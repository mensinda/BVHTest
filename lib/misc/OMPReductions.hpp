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

#include <cstdint>

namespace BVHTest::misc {

struct OMP_fi {
  float    val;
  uint32_t ind;
};

struct OMP_di {
  double   val;
  uint64_t ind;
};

template <class T>
inline const T &omp_red_max(const T &a, const T &b) {
  return a.val > b.val ? a : b;
}

template <class T>
inline const T &omp_red_min(const T &a, const T &b) {
  return a.val < b.val ? a : b;
}

#pragma omp declare reduction(maxValF : OMP_fi : omp_out = omp_red_max <OMP_fi>(omp_out, omp_in))
#pragma omp declare reduction(maxValD : OMP_di : omp_out = omp_red_max <OMP_di>(omp_out, omp_in))

#pragma omp declare reduction(minValF : OMP_fi : omp_out = omp_red_min <OMP_fi>(omp_out, omp_in))
#pragma omp declare reduction(minValD : OMP_di : omp_out = omp_red_min <OMP_di>(omp_out, omp_in))

} // namespace BVHTest::misc
