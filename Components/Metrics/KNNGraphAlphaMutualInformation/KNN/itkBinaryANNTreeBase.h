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
#ifndef itkBinaryANNTreeBase_h
#define itkBinaryANNTreeBase_h

#include "itkBinaryTreeBase.h"
#include <ANN/ANN.h> // ANN declarations

namespace itk
{

/**
 * \class BinaryANNTreeBase
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT BinaryANNTreeBase : public BinaryTreeBase<TListSample>
{
public:
  /** Standard itk. */
  typedef BinaryANNTreeBase           Self;
  typedef BinaryTreeBase<TListSample> Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

  /** ITK type info. */
  itkTypeMacro(BinaryANNTreeBase, BinaryTreeBase);

  /** Typedefs from Superclass. */
  typedef typename Superclass::SampleType                 SampleType;
  typedef typename Superclass::MeasurementVectorType      MeasurementVectorType;
  typedef typename Superclass::MeasurementVectorSizeType  MeasurementVectorSizeType;
  typedef typename Superclass::TotalAbsoluteFrequencyType TotalAbsoluteFrequencyType;

  /** Typedef */
  typedef ANNpointSet ANNPointSetType;

  /** Get the ANN tree. */
  virtual ANNPointSetType *
  GetANNTree(void) const = 0;

protected:
  /** Constructor. */
  BinaryANNTreeBase();

  /** Destructor. */
  ~BinaryANNTreeBase() override = default;

private:
  BinaryANNTreeBase(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBinaryANNTreeBase.hxx"
#endif

#endif // end #ifndef itkBinaryANNTreeBase_h
