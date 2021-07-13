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
#ifndef itkBinaryTreeBase_h
#define itkBinaryTreeBase_h

#include "itkDataObject.h"

namespace itk
{

/**
 * \class BinaryTreeBase
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT BinaryTreeBase : public DataObject
{
public:
  /** Standard itk. */
  typedef BinaryTreeBase           Self;
  typedef DataObject               Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** ITK type info. */
  itkTypeMacro(BinaryTreeBase, DataObject);

  /** Typedef's. */
  typedef TListSample SampleType;

  /** Typedef's. */
  typedef typename SampleType::MeasurementVectorType      MeasurementVectorType;
  typedef typename SampleType::MeasurementVectorSizeType  MeasurementVectorSizeType;
  typedef typename SampleType::TotalAbsoluteFrequencyType TotalAbsoluteFrequencyType;

  /** Set and get the samples: the array of points. */
  itkSetObjectMacro(Sample, SampleType);
  itkGetConstObjectMacro(Sample, SampleType);

  /** Get the number of data points. */
  TotalAbsoluteFrequencyType
  GetNumberOfDataPoints(void) const;

  /** Get the actual number of data points. */
  TotalAbsoluteFrequencyType
  GetActualNumberOfDataPoints(void) const;

  /** Get the dimension of the input data. */
  MeasurementVectorSizeType
  GetDataDimension(void) const;

  /** Generate the tree. */
  virtual void
  GenerateTree(void) = 0;

protected:
  /** Constructor. */
  BinaryTreeBase();

  /** Destructor. */
  ~BinaryTreeBase() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  BinaryTreeBase(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** Store the samples. */
  typename SampleType::Pointer m_Sample;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBinaryTreeBase.hxx"
#endif

#endif // end #ifndef itkBinaryTreeBase_h
