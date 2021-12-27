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
#ifndef itkListSampleCArray_hxx
#define itkListSampleCArray_hxx

#include "itkListSampleCArray.h"
#include "itkNumericTraits.h"

namespace itk
{
namespace Statistics
{

/**
 * ************************ Constructor *************************
 */

template <class TMeasurementVector, class TInternalValue>
ListSampleCArray<TMeasurementVector, TInternalValue>::ListSampleCArray()
{
  this->m_InternalContainer = nullptr;
  this->m_InternalContainerSize = 0;
  this->m_ActualSize = 0;
} // end Constructor


/**
 * ************************ Destructor *************************
 */
template <class TMeasurementVector, class TInternalValue>
ListSampleCArray<TMeasurementVector, TInternalValue>::~ListSampleCArray()
{
  this->DeallocateInternalContainer();
} // end Destructor


/**
 * ************************ GetMeasurementVector *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::GetMeasurementVector(InstanceIdentifier      id,
                                                                           MeasurementVectorType & mv) const
{
  if (id < this->m_InternalContainerSize)
  {
    mv = MeasurementVectorType(this->m_InternalContainer[id], this->GetMeasurementVectorSize(), false);
    return;
  }
  itkExceptionMacro(<< "The requested index is larger than the container size.");

} // end GetMeasurementVector()


/**
 * ************************ GetMeasurementVector *************************
 */

template <class TMeasurementVector, class TInternalValue>
auto
ListSampleCArray<TMeasurementVector, TInternalValue>::GetMeasurementVector(InstanceIdentifier id) const
  -> const MeasurementVectorType &
{
  if (id < this->m_InternalContainerSize)
  {
    this->m_TemporaryMeasurementVector =
      MeasurementVectorType(this->m_InternalContainer[id], this->GetMeasurementVectorSize(), false);
    return this->m_TemporaryMeasurementVector;
  }
  itkExceptionMacro(<< "The requested index is larger than the container size.");

  /** dummy return; */
  return this->m_TemporaryMeasurementVector;

} // end GetMeasurementVector()

/**
 * ************************ SetMeasurement *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::SetMeasurement(InstanceIdentifier      id,
                                                                     unsigned int            dim,
                                                                     const MeasurementType & value)
{
  if (id < this->m_InternalContainerSize)
  {
    this->m_InternalContainer[id][dim] = value;
  }
} // end SetMeasurement()


/**
 * ************************ SetMeasurementVector *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::SetMeasurementVector(InstanceIdentifier            id,
                                                                           const MeasurementVectorType & mv)
{
  unsigned int dim = this->GetMeasurementVectorSize();
  if (id < this->m_InternalContainerSize)
  {
    for (unsigned int i = 0; i < dim; ++i)
    {
      this->m_InternalContainer[id][i] = mv[i];
    }
    // or maybe with iterators
  }
} // end SetMeasurementVector()


/**
 * ************************ Resize *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::Resize(unsigned long size)
{
  /** Resizing deallocates and then allocates memory.
   * Therefore, the memory contains junk just after calling
   * this function. So the m_ActualSize is zero.
   */
  this->m_ActualSize = 0;
  if (this->m_InternalContainer)
  {
    this->DeallocateInternalContainer();
    this->m_InternalContainerSize = 0;
    this->Modified();
  }
  if (size > 0)
  {
    unsigned int dim = this->GetMeasurementVectorSize();
    this->AllocateInternalContainer(size, dim);
    this->m_InternalContainerSize = size;
    this->Modified();
  }

} // end Resize()


/**
 * ************************ SetActualSize *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::SetActualSize(unsigned long size)
{
  if (this->m_ActualSize != size)
  {
    this->m_ActualSize = size;
    this->Modified();
  }
} // end SetActualSize()


/**
 * ************************ GetActualSize *************************
 */

template <class TMeasurementVector, class TInternalValue>
unsigned long
ListSampleCArray<TMeasurementVector, TInternalValue>::GetActualSize()
{
  return this->m_ActualSize;
} // end GetActualSize()


/**
 * ************************ Clear *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::Clear()
{
  this->Resize(0);
} // end Clear()


/**
 * ************************ AllocateInternalContainer *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::AllocateInternalContainer(unsigned long size, unsigned int dim)
{
  this->m_InternalContainer = new InternalDataType[size];
  InternalDataType p = new InternalValueType[size * dim];
  for (unsigned long i = 0; i < size; ++i)
  {
    this->m_InternalContainer[i] = &(p[i * dim]);
  }
} // end AllocateInternalContainer()


/**
 * ************************ DeallocateInternalContainer *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::DeallocateInternalContainer()
{
  if (this->m_InternalContainer)
  {
    delete[] this->m_InternalContainer[0];
    delete[] this->m_InternalContainer;
    this->m_InternalContainer = nullptr;
  }
} // end DeallocateInternalContainer()


/**
 * ************************ GetFrequency *************************
 */

template <class TMeasurementVector, class TInternalValue>
auto
ListSampleCArray<TMeasurementVector, TInternalValue>::GetFrequency(InstanceIdentifier id) const -> AbsoluteFrequencyType
{
  if (id < this->m_InternalContainerSize)
  {
    return itk::NumericTraits<AbsoluteFrequencyType>::One;
  }
  else
  {
    return itk::NumericTraits<AbsoluteFrequencyType>::Zero;
  }
} // end GetFrequency()


/**
 * ************************ PrintSelf *************************
 */

template <class TMeasurementVector, class TInternalValue>
void
ListSampleCArray<TMeasurementVector, TInternalValue>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Internal Data Container: " << &m_InternalContainer << std::endl;

} // end PrintSelf()


} // end of namespace Statistics
} // end of namespace itk

#endif // end itkListSampleCArray_hxx
