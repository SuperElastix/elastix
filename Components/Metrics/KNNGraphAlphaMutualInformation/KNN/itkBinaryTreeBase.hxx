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
#ifndef itkBinaryTreeBase_hxx
#define itkBinaryTreeBase_hxx

#include "itkBinaryTreeBase.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TListSample>
BinaryTreeBase<TListSample>::BinaryTreeBase()
{
  this->m_Sample = nullptr;
} // end Constructor


/**
 * ************************ GetNumberOfDataPoints *************************
 */

template <class TListSample>
auto
BinaryTreeBase<TListSample>::GetNumberOfDataPoints() const -> TotalAbsoluteFrequencyType
{
  if (this->m_Sample)
  {
    return this->m_Sample->GetTotalFrequency();
  }
  return NumericTraits<TotalAbsoluteFrequencyType>::Zero;

} // end GetNumberOfDataPoints()


/**
 * ************************ GetActualNumberOfDataPoints *************************
 */

template <class TListSample>
auto
BinaryTreeBase<TListSample>::GetActualNumberOfDataPoints() const -> TotalAbsoluteFrequencyType
{
  if (this->m_Sample)
  {
    return this->m_Sample->GetActualSize();
  }
  return NumericTraits<TotalAbsoluteFrequencyType>::Zero;

} // end GetActualNumberOfDataPoints()


/**
 * ************************ GetDataDimension *************************
 */

template <class TListSample>
auto
BinaryTreeBase<TListSample>::GetDataDimension() const -> MeasurementVectorSizeType
{
  if (this->m_Sample)
  {
    return this->m_Sample->GetMeasurementVectorSize();
  }
  return NumericTraits<MeasurementVectorSizeType>::Zero;

} // end GetDataDimension()


/*
 * ****************** PrintSelf ******************
 */

template <class TListSample>
void
BinaryTreeBase<TListSample>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Sample: " << this->m_Sample.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkBinaryTreeBase_hxx
