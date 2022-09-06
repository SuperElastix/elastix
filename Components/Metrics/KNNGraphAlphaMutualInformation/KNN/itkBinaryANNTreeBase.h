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
  ITK_DISALLOW_COPY_AND_MOVE(BinaryANNTreeBase);

  /** Standard itk. */
  using Self = BinaryANNTreeBase;
  using Superclass = BinaryTreeBase<TListSample>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** ITK type info. */
  itkTypeMacro(BinaryANNTreeBase, BinaryTreeBase);

  /** Typedefs from Superclass. */
  using typename Superclass::SampleType;
  using typename Superclass::MeasurementVectorType;
  using typename Superclass::MeasurementVectorSizeType;
  using typename Superclass::TotalAbsoluteFrequencyType;

  /** Typedef */
  using ANNPointSetType = ANNpointSet;

  /** Get the ANN tree. */
  virtual ANNPointSetType *
  GetANNTree() const = 0;

protected:
  /** Constructor. */
  BinaryANNTreeBase() = default;

  /** Destructor. */
  ~BinaryANNTreeBase() override = default;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBinaryANNTreeBase.hxx"
#endif

#endif // end #ifndef itkBinaryANNTreeBase_h
