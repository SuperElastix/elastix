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
#ifndef itkListSampleCArray_h
#define itkListSampleCArray_h

#include "itkObjectFactory.h"
//#include "itkListSampleBase.h"
#include "itkSample.h"

namespace itk
{
namespace Statistics
{

/**
 * \class ListSampleCArray
 *
 * \brief A ListSampleBase that internally uses a CArray, which can be accessed
 *
 * This class is useful if some function expects a c-array, but you would
 * like to keep things as much as possible in the itk::Statistics-framework.
 *
 * \todo: the second template argument should be removed, since the GetMeasurementVector
 * method is incorrect when TMeasurementVector::ValueType != TInternalValue.
 *
 * \ingroup Miscellaneous
 */

template <class TMeasurementVector, class TInternalValue = typename TMeasurementVector::ValueType>
class ITK_TEMPLATE_EXPORT ListSampleCArray : public Sample<TMeasurementVector>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ListSampleCArray);

  /** Standard itk. */
  using Self = ListSampleCArray;
  using Superclass = Sample<TMeasurementVector>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory.*/
  itkNewMacro(Self);

  /** ITK type info */
  itkTypeMacro(ListSampleCArray, Sample);

  /** Typedef's from Superclass. */
  using typename Superclass::MeasurementVectorType;
  using typename Superclass::MeasurementVectorSizeType;
  using typename Superclass::MeasurementType;
  using typename Superclass::AbsoluteFrequencyType;
  using typename Superclass::TotalAbsoluteFrequencyType;
  using typename Superclass::InstanceIdentifier;

  /** Typedef's for the internal data container. */
  using InternalValueType = TInternalValue;
  using InternalDataType = InternalValueType *;
  using InternalDataContainerType = InternalDataType *;

  /** Macro to get the internal data container. */
  itkGetConstMacro(InternalContainer, InternalDataContainerType);

  /** Function to resize the data container. */
  void
  Resize(unsigned long n);

  /** Function to set the actual (not the allocated) size of the data container. */
  void
  SetActualSize(unsigned long n);

  /** Function to get the actual (not the allocated) size of the data container. */
  unsigned long
  GetActualSize();

  /** Function to clear the data container. */
  void
  Clear();

  /** Function to get the size of the data container. */
  InstanceIdentifier
  Size() const override
  {
    return this->m_InternalContainerSize;
  }


  /** Function to get a point from the data container.
   * NB: the reference to the returned value remains only valid until the next
   * call to this function.
   * The method GetMeasurementVector( const InstanceIdentifier &id, MeasurementVectorType & mv)
   * is actually a preferred way to get a measurement vector.
   */
  const MeasurementVectorType &
  GetMeasurementVector(InstanceIdentifier id) const override;

  /** Function to get a point from the data container. */
  void
  GetMeasurementVector(InstanceIdentifier id, MeasurementVectorType & mv) const;

  /** Function to set part of a point (measurement) in the data container. */
  void
  SetMeasurement(InstanceIdentifier id, unsigned int dim, const MeasurementType & value);

  /** Function to set a point (measurement vector) in the data container. */
  void
  SetMeasurementVector(InstanceIdentifier id, const MeasurementVectorType & mv);

  /** Function to get the frequency of point i. 1.0 if it exist, 0.0 otherwise. */
  AbsoluteFrequencyType
  GetFrequency(InstanceIdentifier id) const override;

  /** Function to get the total frequency. */
  TotalAbsoluteFrequencyType
  GetTotalFrequency() const override
  {
    return static_cast<TotalAbsoluteFrequencyType>(this->m_InternalContainerSize);
  }


protected:
  ListSampleCArray();
  ~ListSampleCArray() override;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** The internal storage of the data in a C array. */
  InternalDataContainerType m_InternalContainer;
  InstanceIdentifier        m_InternalContainerSize;
  InstanceIdentifier        m_ActualSize;

  /** Dummy needed for GetMeasurementVector(). */
  mutable MeasurementVectorType m_TemporaryMeasurementVector;

  /** Function to allocate the memory of the data container. */
  void
  AllocateInternalContainer(unsigned long size, unsigned int dim);

  /** Function to deallocate the memory of the data container. */
  void
  DeallocateInternalContainer();
};

} // end namespace Statistics
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkListSampleCArray.hxx"
#endif

#endif // end #ifndef itkListSampleCArray_h
