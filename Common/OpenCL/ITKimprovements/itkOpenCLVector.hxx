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
#ifndef itkOpenCLVector_hxx
#define itkOpenCLVector_hxx

#include "itkOpenCLVector.h"

namespace itk
{

//------------------------------------------------------------------------------
template <typename T>
OpenCLVector<T>::OpenCLVector()
  : OpenCLVectorBase(sizeof(T))
{}

//------------------------------------------------------------------------------
template <typename T>
OpenCLVector<T>::OpenCLVector(OpenCLContext * context, const OpenCLMemoryObject::Access access, const std::size_t size)
  : OpenCLVectorBase(sizeof(T))
{
  OpenCLVectorBase::Create(context, access, size);
}


//------------------------------------------------------------------------------
template <typename T>
OpenCLVector<T>::OpenCLVector(const OpenCLVector<T> & other)
  : OpenCLVectorBase(sizeof(T), other)
{}

//------------------------------------------------------------------------------
template <typename T>
OpenCLVector<T>::~OpenCLVector()
{}

//------------------------------------------------------------------------------
template <typename T>
OpenCLVector<T> &
OpenCLVector<T>::operator=(const OpenCLVector<T> & other)
{
  this->Assign(other);
  return *this;
}


//------------------------------------------------------------------------------
template <typename T>
bool
OpenCLVector<T>::IsNull() const
{
  return this->d_ptr == 0;
}


//------------------------------------------------------------------------------
template <typename T>
void
OpenCLVector<T>::Release()
{
  OpenCLVectorBase::Release();
}


//------------------------------------------------------------------------------
template <typename T>
T & OpenCLVector<T>::operator[](const std::size_t index)
{
  itkAssertOrThrowMacro((index < this->m_Size), "OpenCLVector<T>::operator[" << index << "] index out of range");
  if (!this->m_Mapped)
  {
    this->Map();
  }
  return (reinterpret_cast<T *>(this->m_Mapped))[index];
}


//------------------------------------------------------------------------------
template <typename T>
const T & OpenCLVector<T>::operator[](const std::size_t index) const
{
  itkAssertOrThrowMacro((index < this->m_Size), "OpenCLVector<T>::operator[" << index << "] index out of range");
  if (!this->m_Mapped)
  {
    const_cast<OpenCLVector<T> *>(this)->map();
  }
  return (reinterpret_cast<T *>(this->m_Mapped))[index];
}


//------------------------------------------------------------------------------
template <typename T>
void
OpenCLVector<T>::Write(const T * data, const std::size_t count, const std::size_t offset /*= 0 */)
{
  itkAssertOrThrowMacro(((offset + count) <= this->m_Size),
                        "OpenCLVector<T>::Write(data, " << count << ", " << offset
                                                        << ") (offset + count) is out of range")

    OpenCLVectorBase::Write(data, count * sizeof(T), offset * sizeof(T));
}


//------------------------------------------------------------------------------
template <typename T>
void
OpenCLVector<T>::Read(T * data, const std::size_t count, const std::size_t offset /*= 0 */)
{
  itkAssertOrThrowMacro(((offset + count) <= this->m_Size),
                        "OpenCLVector<T>::Read(data, " << count << ", " << offset
                                                       << ") (offset + count) is out of range")

    OpenCLVectorBase::Read(data, count * sizeof(T), offset * sizeof(T));
}


//------------------------------------------------------------------------------
template <typename T>
void
OpenCLVector<T>::Write(const Vector<T> & data, const std::size_t offset /*= 0 */)
{
  this->Write(data.GetDataPointer(), data.GetLength(), offset);
}


//------------------------------------------------------------------------------
template <typename T>
OpenCLContext *
OpenCLVector<T>::GetContext() const
{
  return OpenCLVectorBase::GetContext();
}


//------------------------------------------------------------------------------
template <typename T>
OpenCLBuffer
OpenCLVector<T>::GetBuffer() const
{
  cl_mem id = OpenCLVectorBase::GetMemoryId();

  if (id)
  {
    clRetainMemObject(id);
    return OpenCLBuffer(this->GetContext(), id);
  }
  else
  {
    return OpenCLBuffer();
  }
}


} // namespace itk

#endif // itkOpenCLVector_hxx
