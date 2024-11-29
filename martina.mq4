//+------------------------------------------------------------------+
//|                                                  martina4.mq4     |
//|     EA to call /prices and /chat APIs and place orders           |
//+------------------------------------------------------------------+

#property strict

input int INTERVAL_MINUTES = 5;   // Interval in minutes
datetime lastTriggerTime = 0;

// API endpoints
string pricesApiUrl = "https://gibbor-bot-8cefc426f6ce.herokuapp.com/prices";  // Replace with your actual API URL
string chatApiUrl = "https://gibbor-bot-8cefc426f6ce.herokuapp.com/chat";      // Replace with your actual API URL

// Include JSON library (Ensure this library is included in MQL4\Include folder)
#include <json.mqh>

int OnInit()
{
    Print("EA Loaded: martina4");
    lastTriggerTime = TimeCurrent();
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    datetime currentTime = TimeCurrent();

    // Check if enough time has passed since the last call
    if (currentTime - lastTriggerTime >= INTERVAL_MINUTES * 60)
    {
        lastTriggerTime = currentTime;

        // Prepare the JSON data
        string symbol = Symbol();
        double rawPrice = MarketInfo(symbol, MODE_BID);
        string price = DoubleToString(rawPrice, Digits); // Format price to appropriate decimals

        // Generate the ISO 8601 timestamp
        string timestamp = GetISO8601Timestamp(currentTime);

        // Debugging: Print the generated timestamp
        Print("Generated Timestamp: ", timestamp);

        // Construct JSON payload for /prices endpoint
        string jsonData = StringFormat(
            "{\"symbol\":\"%s\",\"value\":\"%s\",\"timestamp\":\"%s\"}",
            symbol,
            price,
            timestamp
        );

        // Debugging: Print JSON Data
        Print("JSON Data Sent to /prices: ", jsonData);

        // HTTP headers
        string headers = "Content-Type: application/json\r\n";

        // Send price data to /prices endpoint
        string priceResponse = SendHttpPostRequest(pricesApiUrl, headers, jsonData);

        if (priceResponse != "")
        {
            // Prepare JSON data for /chat endpoint
            string chatRequestData = StringFormat("{\"symbol\":\"%s\"}", symbol);

            // Debugging: Print JSON Data for /chat
            Print("JSON Data Sent to /chat: ", chatRequestData);

            // Call /chat endpoint to get trading signal
            string chatResponse = SendHttpPostRequest(chatApiUrl, headers, chatRequestData);

            if (chatResponse != "")
            {
                // Process the chat response and place an entry order
                ProcessChatResponse(chatResponse);
            }
        }
    }
}

// Function to generate ISO 8601 timestamp
string GetISO8601Timestamp(datetime time)
{
    string dateStr = StringFormat("%04d-%02d-%02d", TimeYear(time), TimeMonth(time), TimeDay(time));
    string timeStr = StringFormat("%02d:%02d:%02d", TimeHour(time), TimeMinute(time), TimeSeconds(time));
    return dateStr + "T" + timeStr + "Z";
}

// Function to send HTTP POST request
string SendHttpPostRequest(string url, string headers, string jsonData)
{
    char response[];
    string error_message;

    // Convert JSON string to char array
    char post_data[];
    StringToCharArray(jsonData, post_data);

    // Perform the HTTP POST request
    int result = WebRequest(
        "POST",
        url,
        headers,
        10000,          // Timeout in milliseconds
        post_data,
        response,
        error_message
    );

    // Check the result
    if (result == -1)
    {
        // Print the error message
        Print("API call to ", url, " failed. Error: ", error_message);
        return "";
    }
    else
    {
        // Convert the response buffer to a string
        string responseText = CharArrayToString(response, 0, ArraySize(response));
        Print("API call to ", url, " successful. Response: ", responseText);
        return responseText;
    }
}

// Function to process the chat response and place an entry order
void ProcessChatResponse(string chatResponse)
{
    // Create a new JSON parser instance
    JSONParser parser;
    JSONValue* jsonResponse = parser.parse(chatResponse);

    if (jsonResponse == NULL)
    {
        Print("Failed to parse JSON response. Error: ", parser.getErrorMessage());
        return;
    }

    // Ensure the root is an object
    if (!jsonResponse.isObject())
    {
        Print("Invalid JSON response format.");
        delete jsonResponse;
        return;
    }

    JSONObject* rootObject = (JSONObject*)jsonResponse;

    // Retrieve the "status" value safely
    string status;
    if (!rootObject.getString("status", status))
    {
        Print("Status field missing or not a string.");
        delete jsonResponse;
        return;
    }

    if (status == "success")
    {
        // Retrieve the "data" object
        JSONValue* dataValue = rootObject.getValue("data");
        if (dataValue == NULL || !dataValue.isObject())
        {
            Print("Data field missing or not an object.");
            delete jsonResponse;
            return;
        }

        JSONObject* dataObject = (JSONObject*)dataValue;

        // Retrieve values from the "data" object safely
        string action, entryStr, slStr, tpStr;
        if (!dataObject.getString("action", action) ||
            !dataObject.getString("entry", entryStr) ||
            !dataObject.getString("sl", slStr) ||
            !dataObject.getString("tp", tpStr))
        {
            Print("Incomplete data in JSON response.");
            delete jsonResponse;
            return;
        }

        double entryPrice = StringToDouble(entryStr);
        double stopLoss = StringToDouble(slStr);
        double takeProfit = StringToDouble(tpStr);

        // Determine the order type
        int orderType;
        if (action == "buy-limit")
            orderType = OP_BUYLIMIT;
        else if (action == "sell-limit")
            orderType = OP_SELLLIMIT;
        else if (action == "buy-stop")
            orderType = OP_BUYSTOP;
        else if (action == "sell-stop")
            orderType = OP_SELLSTOP;
        else
        {
            Print("Unknown action type: ", action);
            delete jsonResponse;
            return;
        }

        // Place the pending order
        int ticket = OrderSend(
            Symbol(),          // symbol
            orderType,         // order type
            0.1,               // lots (adjust as needed)
            entryPrice,        // price
            3,                 // slippage (adjust as needed)
            stopLoss,          // stop loss
            takeProfit,        // take profit
            "Entry Order",     // comment
            0,                 // magic number (set if needed)
            0,                 // expiration
            clrNONE            // arrow color
        );

        if (ticket > 0)
        {
            Print("Order placed successfully. Ticket#: ", ticket);
        }
        else
        {
            Print("Order placement failed. Error: ", GetLastError());
        }
    }
    else
    {
        Print("Chat API response status not success: ", status);
    }

    // Clean up
    delete jsonResponse;
}
