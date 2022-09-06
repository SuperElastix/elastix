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
  ITK_DISALLOW_COPY_AND_MOVE(BinaryTreeBase);

  /** Standard itk. */
  using Self = BinaryTreeBase;
  using Superclass = DataObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** ITK type info. */
  itkTypeMacro(BinaryTreeBase, DataObject);

  /** Typedef's. */
  using SampleType = TListSample;

  /** Typedef's. */
  using MeasurementVectorType = typename SampleType::MeasurementVectorType;
  using MeasurementVectorSizeType = typename SampleType::MeasurementVectorSizeType;
  using TotalAbsoluteFrequencyType = typename SampleType::TotalAbsoluteFrequencyType;

  /** Set and get the samples: the array of points. */
  itkSetObjectMacro(Sample, SampleType);
  itkGetConstObjectMacro(Sample, SampleType);

  /** Get the number of data points. */
  TotalAbsoluteFrequencyType
  GetNumberOfDataPoints() const;

  /** Get the actual number of data points. */
  TotalAbsoluteFrequencyType
  GetActualNumberOfDataPoints() const;

  /** Get the dimension of the input data. */
  MeasurementVectorSizeType
  GetDataDimension() const;

  /** Generate the tree. */
  virtual void
  GenerateTree() = 0;

protected:
  /** Constructor. */
  BinaryTreeBase();

  /** Destructor. */
  ~BinaryTreeBase() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Store the samples. */
  typename SampleType::Pointer m_Sample;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBinaryTreeBase.hxx"
#endif

#endif // end #ifndef itkBinaryTreeBase_h
