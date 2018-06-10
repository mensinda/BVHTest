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

/*
 * The algorithms here are copy pasted from the C++ algorithms library
 */

#pragma once

#ifdef __CUDACC__
#  ifndef CUDA_CALL
#    define CUDA_CALL __device__ __forceinline__
#  endif
#else
#  ifndef CUDA_CALL
#    define CUDA_CALL inline
#  endif
#endif

template <typename _Iterator1, typename _Iterator2>
CUDA_CALL bool CUDA_compareIterIter(_Iterator1 __it1, _Iterator2 __it2) {
  return *__it1 < *__it2;
}

template <typename _Iterator, typename _Value>
CUDA_CALL bool CUDA_compareIterVal(_Iterator __it, _Value &__val) {
  return *__it < __val;
}


template <typename _Tp>
CUDA_CALL void CUDA___push_heap(_Tp *__first, long int __holeIndex, long int __topIndex, _Tp __value) {
  long int __parent = (__holeIndex - 1) / 2;
  while (__holeIndex > __topIndex && CUDA_compareIterVal(__first + __parent, __value)) {
    *(__first + __holeIndex) = static_cast<_Tp &&>(*(__first + __parent));
    __holeIndex              = __parent;
    __parent                 = (__holeIndex - 1) / 2;
  }
  *(__first + __holeIndex) = static_cast<_Tp &&>(__value);
}

template <typename _Tp>
CUDA_CALL void CUDA__adjust_heap(_Tp *__first, long int __holeIndex, long int __len, _Tp __value) {
  const long int __topIndex    = __holeIndex;
  long int       __secondChild = __holeIndex;
  while (__secondChild < (__len - 1) / 2) {
    __secondChild = 2 * (__secondChild + 1);
    if (CUDA_compareIterIter(__first + __secondChild, __first + (__secondChild - 1))) __secondChild--;
    *(__first + __holeIndex) = static_cast<_Tp &&>(*(__first + __secondChild));
    __holeIndex              = __secondChild;
  }
  if ((__len & 1) == 0 && __secondChild == (__len - 2) / 2) {
    __secondChild            = 2 * (__secondChild + 1);
    *(__first + __holeIndex) = static_cast<_Tp &&>(*(__first + (__secondChild - 1)));
    __holeIndex              = __secondChild - 1;
  }
  CUDA___push_heap(__first, __holeIndex, __topIndex, static_cast<_Tp &&>(__value));
}

template <typename _Tp>
CUDA_CALL void CUDA__pop_heap(_Tp *__first, _Tp *__last, _Tp *__result) {
  _Tp __value = static_cast<_Tp &&>(*__result);
  *__result   = static_cast<_Tp &&>(*__first);
  CUDA__adjust_heap(__first, 0, __last - __first, static_cast<_Tp &&>(__value));
}

/**
 *  @brief  Pop an element off a heap.
 *  @param  __first  Start of heap.
 *  @param  __last   End of heap.
 *  @pre    [__first, __last) is a valid, non-empty range.
 *  @ingroup heap_algorithms
 *
 *  This operation pops the top of the heap.  The elements __first
 *  and __last-1 are swapped and [__first,__last-1) is made into a
 *  heap.
 */
template <typename _Tp>
CUDA_CALL void CUDA_pop_heap(_Tp *__first, _Tp *__last) {
  if (__last - __first > 1) {
    --__last;
    CUDA__pop_heap(__first, __last, __last);
  }
}

/**
 *  @brief  Push an element onto a heap.
 *  @param  __first  Start of heap.
 *  @param  __last   End of heap + element.
 *  @ingroup heap_algorithms
 *
 *  This operation pushes the element at last-1 onto the valid heap
 *  over the range [__first,__last-1).  After completion,
 *  [__first,__last) is a valid heap.
 */
template <typename _Tp>
CUDA_CALL void CUDA_push_heap(_Tp *__first, _Tp *__last) {
  CUDA___push_heap(__first, (__last - __first) - 1, 0, static_cast<_Tp &&>(*(__last - 1)));
}
