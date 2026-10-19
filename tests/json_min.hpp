#pragma once
// A small JSON reader for config files and the case manifest: objects, numbers,
// strings, booleans, and arrays. Not a general JSON parser.
// Numbers are kept as the original text and converted with strtod/strtoll on request, so
// there is no rounding other than the conversion itself.
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace jsonmin {

struct Value {
    enum Kind { Num, Str, Bool, Arr, Obj } kind = Num;
    std::string text;            // Num: source text; Str: decoded string
    bool b = false;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;
    double num() const { return std::strtod(text.c_str(), nullptr); }
    long long integer() const {
        char* end = nullptr;
        long long v = std::strtoll(text.c_str(), &end, 10);
        if (!end || *end) throw std::runtime_error("not an integer: " + text);
        return v;
    }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}
    std::map<std::string, Value> object() {
        std::map<std::string, Value> o;
        ws(); expect('{'); ws();
        if (peek() == '}') { ++i_; return o; }
        for (;;) {
            ws(); std::string k = string(); ws(); expect(':'); ws();
            if (o.count(k)) throw std::runtime_error("duplicate key " + k);
            o[k] = value(); ws();
            if (peek() == ',') { ++i_; continue; }
            expect('}'); break;
        }
        ws();
        if (depth_ == 0 && i_ != s_.size()) throw std::runtime_error("trailing text after the object");
        return o;
    }
private:
    const std::string& s_;
    size_t i_ = 0;
    size_t depth_ = 0;
    char peek() const { if (i_ >= s_.size()) throw std::runtime_error("unexpected end"); return s_[i_]; }
    void expect(char c) { if (peek() != c) throw std::runtime_error(std::string("expected '") + c + "'"); ++i_; }
    void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\t' || s_[i_] == '\r')) ++i_; }
    std::string string() {
        expect('"'); std::string o;
        while (peek() != '"') {
            char c = s_[i_++];
            if (c == '\\') {
                char e = s_[i_++];
                if (e == 'n') o += '\n'; else if (e == 't') o += '\t'; else if (e == '"' || e == '\\' || e == '/') o += e;
                else throw std::runtime_error("unsupported escape");
            } else o += c;
        }
        ++i_; return o;
    }
    Value value() {
        Value v;
        char c = peek();
        if (c == '{') { v.kind = Value::Obj; ++depth_; v.obj = object(); --depth_; }
        else if (c == '"') { v.kind = Value::Str; v.text = string(); }
        else if (c == '[') {
            v.kind = Value::Arr; ++i_; ws();
            if (peek() == ']') { ++i_; return v; }
            for (;;) { ws(); v.arr.push_back(value()); ws(); if (peek() == ',') { ++i_; continue; } expect(']'); break; }
        } else if (s_.compare(i_, 4, "true") == 0) { v.kind = Value::Bool; v.b = true; i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v.kind = Value::Bool; v.b = false; i_ += 5; }
        else {
            size_t j = i_;
            while (j < s_.size() && (std::isdigit((unsigned char)s_[j]) || s_[j]=='-' || s_[j]=='+' || s_[j]=='.' || s_[j]=='e' || s_[j]=='E')) ++j;
            if (j == i_) throw std::runtime_error("bad value");
            v.kind = Value::Num; v.text = s_.substr(i_, j - i_); i_ = j;
        }
        return v;
    }
};

} // namespace jsonmin
