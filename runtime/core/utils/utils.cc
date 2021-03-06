// Copyright (c) 2021 Mobvoi Inc (Zhendong Peng)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils/utils.h"

#include "utils/log.h"

namespace wenet {

float LogAdd(const float& x, const float& y) {
  static float num_min = -std::numeric_limits<float>::max();
  if (x <= num_min) return y;
  if (y <= num_min) return x;
  float xmax = std::max(x, y);
  return std::log(std::exp(x - xmax) + std::exp(y - xmax)) + xmax;
}

void SplitString(const std::string& str, std::vector<std::string>* strs) {
  SplitStringToVector(str, " \t", true, strs);
}

void SplitStringToVector(const std::string& full, const char* delim,
                         bool omit_empty_strings,
                         std::vector<std::string>* out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

std::string UTF8CodeToUTF8String(int code) {
  std::ostringstream ostr;
  if (code < 0) {
    LOG(ERROR) << "LabelsToUTF8String: Invalid character found: " << code;
    return ostr.str();
  } else if (code < 0x80) {
    ostr << static_cast<char>(code);
  } else if (code < 0x800) {
    ostr << static_cast<char>((code >> 6) | 0xc0);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x10000) {
    ostr << static_cast<char>((code >> 12) | 0xe0);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x200000) {
    ostr << static_cast<char>((code >> 18) | 0xf0);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x4000000) {
    ostr << static_cast<char>((code >> 24) | 0xf8);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else {
    ostr << static_cast<char>((code >> 30) | 0xfc);
    ostr << static_cast<char>(((code >> 24) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  }
  return ostr.str();
}

// Split utf8 string into characters.
bool SplitUTF8String(const std::string& str,
                     std::vector<std::string>* characters) {
  const char* data = str.data();
  const size_t length = str.size();
  for (size_t i = 0; i < length; /* no update */) {
    int c = data[i++] & 0xff;
    if ((c & 0x80) == 0) {
      characters->push_back(UTF8CodeToUTF8String(c));
    } else {
      if ((c & 0xc0) == 0x80) {
        LOG(ERROR) << "UTF8StringToLabels: continuation byte as lead byte";
        return false;
      }
      int count = (c >= 0xc0) + (c >= 0xe0) + (c >= 0xf0) + (c >= 0xf8) +
                  (c >= 0xfc);
      int code = c & ((1 << (6 - count)) - 1);
      while (count != 0) {
        if (i == length) {
          LOG(ERROR) << "UTF8StringToLabels: truncated utf-8 byte sequence";
          return false;
        }
        char cb = data[i++];
        if ((cb & 0xc0) != 0x80) {
          LOG(ERROR) << "UTF8StringToLabels: missing/invalid continuation byte";
          return false;
        }
        code = (code << 6) | (cb & 0x3f);
        count--;
      }
      if (code < 0) {
        // This should not be able to happen.
        LOG(ERROR) << "UTF8StringToLabels: Invalid character found: " << c;
        return false;
      }
      characters->push_back(UTF8CodeToUTF8String(code));
    }
  }
  return true;
}

std::string ProcessBlank(const std::string& str) {
  std::string result;
  if (!str.empty()) {
    std::vector<std::string> characters;
    if (SplitUTF8String(str, &characters)) {
      for (std::string& character : characters) {
        if (character != kSpaceSymbol) {
          result.append(character);
        } else {
          // Ignore consecutive space or located in head
          if (!result.empty() && result.back() != ' ') {
            result.push_back(' ');
          }
        }
      }
      // Ignore tailing space
      if (!result.empty() && result.back() == ' ') {
        result.pop_back();
      }
      for (size_t i = 0; i < result.size(); ++i) {
        result[i] = tolower(result[i]);
      }
    }
  }
  return result;
}
int preNUm(unsigned char byte) {
    unsigned char mask = 0x80;
    int num = 0;
    for (int i = 0; i < 8; i++) {
        if ((byte & mask) == mask) {
            mask = mask >> 1;
            num++;
        } else {
            break;
        }
    }
    return num;
}

bool isUtf8(unsigned char* data, int len) {
    int num = 0;
    int i = 0;
    while (i < len) {
        if ((data[i] & 0x80) == 0x00) {
            // 0XXX_XXXX
            i++;
            continue;
        }
        else if ((num = preNUm(data[i])) > 2) {
        // 110X_XXXX 10XX_XXXX
        // 1110_XXXX 10XX_XXXX 10XX_XXXX
        // 1111_0XXX 10XX_XXXX 10XX_XXXX 10XX_XXXX
        // 1111_10XX 10XX_XXXX 10XX_XXXX 10XX_XXXX 10XX_XXXX
        // 1111_110X 10XX_XXXX 10XX_XXXX 10XX_XXXX 10XX_XXXX 10XX_XXXX
        // preNUm()
        i++;
        for(int j = 0; j < num - 1; j++) {
            //????????num - 1 ????????????????10??
            if ((data[i] & 0xc0) != 0x80) {
                    return false;
                }
                i++;
        }
    } else {
        //not utf-8
        return false;
    }
    }
    return true;
}
void HexStrToByte(const char* source, unsigned char* dest, int sourceLen)
{
    short i;
    unsigned char highByte, lowByte;

    for (i = 0; i < sourceLen; i += 2)
    {
        highByte = toupper(source[i]);
        lowByte = toupper(source[i + 1]);

        if (highByte > 0x39)
            highByte -= 0x37;
        else
            highByte -= 0x30;

        if (lowByte > 0x39)
            lowByte -= 0x37;
        else
            lowByte -= 0x30;

        dest[i / 2] = (highByte << 4) | lowByte;
    }
    return;
}

std::string ProcessUnicodeStr(const std::string &str)
{
    std::vector<std::string> slices;
    std::string result;
    std::string delim = "|";
    SplitStringToVector(str, delim.c_str(), true, &slices);
    for (std::string &s:slices)
    {
        char arrayout[512] = { 0 };
	    HexStrToByte(s.c_str(), (unsigned char*)arrayout, s.size());
	    bool isutf8 = isUtf8((unsigned char*)arrayout, s.size());
	    if (isutf8)
	    {
	        result += arrayout;
	    }
	    else
	    {
	        result += "";
	    }
    }
    return result;

}

}  // namespace wenet
