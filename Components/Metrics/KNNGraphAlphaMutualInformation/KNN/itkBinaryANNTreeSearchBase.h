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
#ifndef itkBinaryANNTreeSearchBase_h
#define itkBinaryANNTreeSearchBase_h

#include "itkBinaryTreeSearchBase.h"
#include "itkBinaryANNTreeBase.h"
#include "ANN/ANN.h"

namespace itk
{

/**
 * \class BinaryANNTreeSearchBase
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT BinaryANNTreeSearchBase : public BinaryTreeSearchBase<TListSample>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BinaryANNTreeSearchBase);

  /** Standard itk. */
  using Self = BinaryANNTreeSearchBase;
  using Superclass = BinaryTreeSearchBase<TListSample>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** ITK type info. */
  itkTypeMacro(BinaryANNTreeSearchBase, BinaryTreeSearchBase);

  /** Typedefs from Superclass. */
  using typename Superclass::ListSampleType;
  using typename Superclass::BinaryTreeType;
  using typename Superclass::BinaryTreePointer;
  using typename Superclass::MeasurementVectorType;
  using typename Superclass::IndexArrayType;
  using typename Superclass::DistanceArrayType;

  /** Typedefs from ANN. */
  using ANNPointType = ANNpoint;             // double *
  using ANNIndexType = ANNidx;               // int
  using ANNIndexArrayType = ANNidxArray;     // int *
  using ANNDistanceType = ANNdist;           // double
  using ANNDistanceArrayType = ANNdistArray; // double *

  /** An itk ANN tree. */
  using BinaryANNTreeType = BinaryANNTreeBase<ListSampleType>;

  /** Set and get the binary tree. */
  void
  SetBinaryTree(BinaryTreeType * tree) override;

  // const BinaryTreeType * GetBinaryTree() const;

protected:
  BinaryANNTreeSearchBase();
  ~BinaryANNTreeSearchBase() override = default;

  /** Member variables. */
  typename BinaryANNTreeType::Pointer m_BinaryTreeAsITKANNType;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBinaryANNTreeSearchBase.hxx"
#endif

#endif // end #ifndef itkBinaryANNTreeSearchBase_h
