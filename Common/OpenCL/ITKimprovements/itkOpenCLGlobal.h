/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkOpenCLGlobal_h
#define itkOpenCLGlobal_h

#include <memory>

namespace itk
{
template <typename T>
static inline T *
OpenCLGetPtrHelper(T * ptr)
{
  return ptr;
}
template <typename TObjectType>
static inline typename std::unique_ptr<TObjectType>::element_type *
OpenCLGetPtrHelper(const std::unique_ptr<TObjectType> & p)
{
  return p.get();
}

#define ITK_OPENCL_DECLARE_PRIVATE(Class)                                                                              \
  inline Class##Pimpl *       d_func() { return reinterpret_cast<Class##Pimpl *>(OpenCLGetPtrHelper(d_ptr)); }         \
  inline const Class##Pimpl * d_func() const                                                                           \
  {                                                                                                                    \
    return reinterpret_cast<const Class##Pimpl *>(OpenCLGetPtrHelper(d_ptr));                                          \
  }                                                                                                                    \
  friend class Class##Pimpl;

#define ITK_OPENCL_DECLARE_PRIVATE_D(Dptr, Class)                                                                      \
  inline Class##Pimpl *       d_func() { return reinterpret_cast<Class##Pimpl *>(Dptr); }                              \
  inline const Class##Pimpl * d_func() const { return reinterpret_cast<const Class##Pimpl *>(Dptr); }                  \
  friend class Class##Pimpl;

#define ITK_OPENCL_DECLARE_PUBLIC(Class)                                                                               \
  inline Class *       q_func() { return static_cast<Class *>(q_ptr); }                                                \
  inline const Class * q_func() const { return static_cast<const Class *>(q_ptr); }                                    \
  friend class Class;

#define ITK_OPENCL_D(Class) Class##Pimpl * const d = d_func()
#define ITK_OPENCL_Q(Class) Class * const q = q_func()
} // end namespace itk

#endif /* itkOpenCLGlobal_h */
