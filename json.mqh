//+------------------------------------------------------------------+
//|                                                  json.mqh        |
//|               JSON Parsing Library for MQL4 (MetaTrader 4)       |
//|                                                                  |
//| This library provides classes and methods to parse JSON strings  |
//| and access JSON data structures in MQL4.                         |
//+------------------------------------------------------------------+

#ifndef __JSON_MQH__
#define __JSON_MQH__

#property strict

enum ENUM_JSON_TYPE { JSON_NULL, JSON_OBJECT, JSON_ARRAY, JSON_NUMBER, JSON_STRING, JSON_BOOL };

// Base class for all JSON values
class JSONValue
{
protected:
    ENUM_JSON_TYPE _type;

public:
    JSONValue() { _type = JSON_NULL; }
    virtual ~JSONValue() {}

    ENUM_JSON_TYPE getType() { return _type; }
    void setType(ENUM_JSON_TYPE t) { _type = t; }

    bool isString() { return _type == JSON_STRING; }
    bool isNull()   { return _type == JSON_NULL; }
    bool isObject() { return _type == JSON_OBJECT; }
    bool isArray()  { return _type == JSON_ARRAY; }
    bool isNumber() { return _type == JSON_NUMBER; }
    bool isBool()   { return _type == JSON_BOOL; }

    virtual string toString() { return ""; }

    // Placeholder methods
    virtual string getString() { return ""; }
    virtual double getDouble() { return 0.0; }
    virtual long   getLong()   { return 0; }
    virtual int    getInt()    { return 0; }
    virtual bool   getBool()   { return false; }
};

// JSON String
class JSONString : public JSONValue
{
private:
    string _string;

public:
    JSONString(string s)
    {
        _string = s;
        setType(JSON_STRING);
    }

    JSONString()
    {
        setType(JSON_STRING);
    }

    string getString() { return _string; }
    string toString() { return "\"" + _string + "\""; }
};

// JSON Number
class JSONNumber : public JSONValue
{
private:
    double _number;

public:
    JSONNumber(double num)
    {
        _number = num;
        setType(JSON_NUMBER);
    }

    double getDouble() { return _number; }
    long   getLong()   { return (long)_number; }
    int    getInt()    { return (int)_number; }

    string toString() { return DoubleToString(_number, 8); }
};

// JSON Boolean
class JSONBool : public JSONValue
{
private:
    bool _bool;

public:
    JSONBool(bool b)
    {
        _bool = b;
        setType(JSON_BOOL);
    }

    bool getBool() { return _bool; }
    string toString() { return _bool ? "true" : "false"; }
};

// JSON Null
class JSONNull : public JSONValue
{
public:
    JSONNull()
    {
        setType(JSON_NULL);
    }

    string toString() { return "null"; }
};

// Forward declarations
class JSONObject;
class JSONArray;

// JSONObject
class JSONObject : public JSONValue
{
private:
    // Key-Value pair struct
    struct KeyValuePair
    {
        string key;
        JSONValue* value;
    };

    KeyValuePair members[];

public:
    JSONObject()
    {
        setType(JSON_OBJECT);
    }

    ~JSONObject()
    {
        // Clean up the values
        for (int i = ArraySize(members) - 1; i >= 0; i--)
        {
            if (members[i].value != NULL)
            {
                delete members[i].value;
                members[i].value = NULL;
            }
        }
    }

    // Add or replace a key-value pair
    void put(string key, JSONValue* v)
    {
        int index = findKeyIndex(key);
        if (index >= 0)
        {
            // Key exists, replace the value
            if (members[index].value != NULL)
                delete members[index].value;
            members[index].value = v;
        }
        else
        {
            // Key doesn't exist, add new entry
            int newSize = ArrayResize(members, ArraySize(members) + 1);
            members[newSize - 1].key = key;
            members[newSize - 1].value = v;
        }
    }

    // Get value by key
    JSONValue* getValue(string key)
    {
        int index = findKeyIndex(key);
        if (index >= 0)
        {
            return members[index].value;
        }
        return NULL;
    }

    // Find the index of a key
    int findKeyIndex(string key)
    {
        for (int i = 0; i < ArraySize(members); i++)
        {
            if (members[i].key == key)
            {
                return i;
            }
        }
        return -1;
    }

    // Getters with two parameters, return bool
    bool getString(string key, string &out)
    {
        JSONValue* value = getValue(key);
        if (value != NULL && value.isString())
        {
            out = value.getString();
            return true;
        }
        return false;
    }

    bool getDouble(string key, double &out)
    {
        JSONValue* value = getValue(key);
        if (value != NULL && value.isNumber())
        {
            out = value.getDouble();
            return true;
        }
        return false;
    }

    bool getLong(string key, long &out)
    {
        JSONValue* value = getValue(key);
        if (value != NULL && value.isNumber())
        {
            out = value.getLong();
            return true;
        }
        return false;
    }

    bool getBool(string key, bool &out)
    {
        JSONValue* value = getValue(key);
        if (value != NULL && value.isBool())
        {
            out = value.getBool();
            return true;
        }
        return false;
    }

    // Convert to string
    string toString()
    {
        string s = "{";
        for (int i = 0; i < ArraySize(members); i++)
        {
            if (i > 0)
                s += ",";
            s += "\"" + members[i].key + "\":" + members[i].value.toString();
        }
        s += "}";
        return s;
    }
};

// JSONArray
class JSONArray : public JSONValue
{
private:
    JSONValue* elements[];

public:
    JSONArray()
    {
        setType(JSON_ARRAY);
    }

    ~JSONArray()
    {
        // Clean up the elements
        for (int i = ArraySize(elements) - 1; i >= 0; i--)
        {
            if (elements[i] != NULL)
            {
                delete elements[i];
                elements[i] = NULL;
            }
        }
    }

    // Add or replace an element at index
    bool put(int index, JSONValue* v)
    {
        if (index >= ArraySize(elements))
        {
            int newSize = ArrayResize(elements, index + 1);
            if (newSize <= index)
                return false;
            // Initialize new elements to NULL
            for (int i = ArraySize(elements) - 1; i >= index; i--)
            {
                elements[i] = NULL;
            }
        }
        // Delete old entry if any
        if (elements[index] != NULL)
            delete elements[index];
        elements[index] = v;
        return true;
    }

    // Get value by index
    JSONValue* getValue(int index)
    {
        if (index >= 0 && index < ArraySize(elements))
        {
            return elements[index];
        }
        return NULL;
    }

    // Getters with two parameters, return bool
    bool getString(int index, string &out)
    {
        JSONValue* value = getValue(index);
        if (value != NULL && value.isString())
        {
            out = value.getString();
            return true;
        }
        return false;
    }

    // Convert to string
    string toString()
    {
        string s = "[";
        for (int i = 0; i < ArraySize(elements); i++)
        {
            if (i > 0)
                s += ",";
            s += elements[i].toString();
        }
        s += "]";
        return s;
    }
};

// JSONParser class
class JSONParser
{
private:
    int _pos;
    ushort _in[];
    int _len;
    string _instr;
    int _errCode;
    string _errMsg;

    void setError(int code = 1, string msg = "unknown error")
    {
        _errCode |= code;
        if (_errMsg == "")
        {
            _errMsg = "JSONParser::Error " + msg;
        }
        else
        {
            _errMsg = _errMsg + "\n" + msg;
        }
    }

    // Helper methods
    void skipWhitespace()
    {
        while (_pos < _len && (_in[_pos] == ' ' || _in[_pos] == '\t' || _in[_pos] == '\n' || _in[_pos] == '\r'))
        {
            _pos++;
        }
    }

    ushort peek()
    {
        if (_pos < _len)
            return _in[_pos];
        else
            return 0;
    }

    ushort next()
    {
        if (_pos < _len)
            return _in[_pos++];
        else
            return 0;
    }

    string parseString()
    {
        string result = "";
        if (next() != '\"')
        {
            setError(1, "Expected '\"' at the beginning of a string");
            return result;
        }
        while (_pos < _len)
        {
            ushort c = next();
            if (c == '\"')
            {
                return result;
            }
            else if (c == '\\')
            {
                if (_pos >= _len)
                {
                    setError(1, "Unexpected end of input in string escape");
                    return result;
                }
                c = next();
                if (c == '\"' || c == '\\' || c == '/')
                    result += ShortToString(c);
                else if (c == 'b')
                    result += ShortToString(8);   // backspace
                else if (c == 'f')
                    result += ShortToString(12);  // form feed
                else if (c == 'n')
                    result += ShortToString(10);  // line feed
                else if (c == 'r')
                    result += ShortToString(13);  // carriage return
                else if (c == 't')
                    result += ShortToString(9);   // tab
                else
                {
                    setError(1, "Invalid escape character in string");
                    return result;
                }
            }
            else
            {
                result += ShortToString(c);
            }
        }
        setError(1, "Unexpected end of input in string");
        return result;
    }

    JSONValue* parseNumber()
    {
        int start = _pos;
        bool hasDecimal = false;
        while (_pos < _len)
        {
            ushort c = _in[_pos];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+')
            {
                _pos++;
            }
            else if (c == '.')
            {
                hasDecimal = true;
                _pos++;
            }
            else
            {
                break;
            }
        }
        string numStr = StringSubstr(_instr, start, _pos - start);
        double num = StringToDouble(numStr);
        return new JSONNumber(num);
    }

    JSONValue* parseValue()
    {
        skipWhitespace();
        ushort c = peek();
        if (c == '\"')
        {
            return new JSONString(parseString());
        }
        else if ((c >= '0' && c <= '9') || c == '-' || c == '+')
        {
            return parseNumber();
        }
        else if (c == '{')
        {
            return parseObject();
        }
        else if (c == '[')
        {
            return parseArray();
        }
        else if (StringSubstr(_instr, _pos, 4) == "true")
        {
            _pos += 4;
            return new JSONBool(true);
        }
        else if (StringSubstr(_instr, _pos, 5) == "false")
        {
            _pos += 5;
            return new JSONBool(false);
        }
        else if (StringSubstr(_instr, _pos, 4) == "null")
        {
            _pos += 4;
            return new JSONNull();
        }
        else
        {
            setError(1, "Invalid value");
            return NULL;
        }
    }

    JSONObject* parseObject()
    {
        if (next() != '{')
        {
            setError(1, "Expected '{' at the beginning of object");
            return NULL;
        }
        JSONObject* obj = new JSONObject();
        skipWhitespace();
        if (peek() == '}')
        {
            next(); // Consume '}'
            return obj;
        }
        while (true)
        {
            skipWhitespace();
            if (peek() != '\"')
            {
                setError(1, "Expected string key in object");
                delete obj;
                return NULL;
            }
            string key = parseString();
            skipWhitespace();
            if (next() != ':')
            {
                setError(1, "Expected ':' after key in object");
                delete obj;
                return NULL;
            }
            JSONValue* value = parseValue();
            if (value == NULL)
            {
                delete obj;
                return NULL;
            }
            obj.put(key, value);
            skipWhitespace();
            ushort c = peek();
            if (c == '}')
            {
                next(); // Consume '}'
                break;
            }
            else if (c == ',')
            {
                next(); // Consume ','
            }
            else
            {
                setError(1, "Expected ',' or '}' in object");
                delete obj;
                return NULL;
            }
        }
        return obj;
    }

    JSONArray* parseArray()
    {
        if (next() != '[')
        {
            setError(1, "Expected '[' at the beginning of array");
            return NULL;
        }
        JSONArray* arr = new JSONArray();
        skipWhitespace();
        if (peek() == ']')
        {
            next(); // Consume ']'
            return arr;
        }
        int index = 0;
        while (true)
        {
            JSONValue* value = parseValue();
            if (value == NULL)
            {
                delete arr;
                return NULL;
            }
            arr.put(index++, value);
            skipWhitespace();
            ushort c = peek();
            if (c == ']')
            {
                next(); // Consume ']'
                break;
            }
            else if (c == ',')
            {
                next(); // Consume ','
            }
            else
            {
                setError(1, "Expected ',' or ']' in array");
                delete arr;
                return NULL;
            }
            skipWhitespace();
        }
        return arr;
    }

public:
    JSONParser()
    {
        _pos = 0;
        _len = 0;
        _errCode = 0;
        _errMsg = "";
    }

    JSONValue* parse(string s)
    {
        _instr = s;
        _len = StringLen(s);
        _pos = 0;
        _errCode = 0;
        _errMsg = "";
        StringToShortArray(s, _in);
        JSONValue* result = parseValue();
        if (_errCode != 0)
        {
            delete result;
            return NULL;
        }
        return result;
    }

    int getErrorCode()
    {
        return _errCode;
    }

    string getErrorMessage()
    {
        return _errMsg;
    }
};

#endif // __JSON_MQH__
