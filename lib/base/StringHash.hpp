/*
 * Copyright (C) 2018 Daniel Mensinger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

namespace BVHTest::base {

const size_t FNV1A_BASE  = 2166136261;
const size_t FNV1A_PRIME = 16777619;

inline size_t fnv1aHash(const char *data) {
  size_t hash = FNV1A_BASE;
  while (*data != 0) {
    hash ^= static_cast<size_t>(*(data++));
    hash *= FNV1A_PRIME;
  }
  return hash;
}

constexpr size_t fnv1aHash(const char *data, size_t n) {
  size_t hash = FNV1A_BASE;
  for (size_t i = 0; i < n; ++i) {
    hash ^= static_cast<size_t>(data[i]);
    hash *= FNV1A_PRIME;
  }
  return hash;
}

size_t fnv1aHash(std::string const &_str) { return fnv1aHash(_str.c_str(), _str.size()); }

constexpr size_t operator"" _h(char const *data, size_t n) { return fnv1aHash(data, n); }

} // namespace BVHTest::base
