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
  /** Standard itk. */
  typedef BinaryANNTreeSearchBase           Self;
  typedef BinaryTreeSearchBase<TListSample> Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** ITK type info. */
  itkTypeMacro(BinaryANNTreeSearchBase, BinaryTreeSearchBase);

  /** Typedefs from Superclass. */
  typedef typename Superclass::ListSampleType        ListSampleType;
  typedef typename Superclass::BinaryTreeType        BinaryTreeType;
  typedef typename Superclass::BinaryTreePointer     BinaryTreePointer;
  typedef typename Superclass::MeasurementVectorType MeasurementVectorType;
  typedef typename Superclass::IndexArrayType        IndexArrayType;
  typedef typename Superclass::DistanceArrayType     DistanceArrayType;

  /** Typedefs from ANN. */
  typedef ANNpoint     ANNPointType;         // double *
  typedef ANNidx       ANNIndexType;         // int
  typedef ANNidxArray  ANNIndexArrayType;    // int *
  typedef ANNdist      ANNDistanceType;      // double
  typedef ANNdistArray ANNDistanceArrayType; // double *

  /** An itk ANN tree. */
  typedef BinaryANNTreeBase<ListSampleType> BinaryANNTreeType;

  /** Set and get the binary tree. */
  void
  SetBinaryTree(BinaryTreeType * tree) override;

  // const BinaryTreeType * GetBinaryTree( void ) const;

protected:
  BinaryANNTreeSearchBase();
  ~BinaryANNTreeSearchBase() override;

  /** Member variables. */
  typename BinaryANNTreeType::Pointer m_BinaryTreeAsITKANNType;

private:
  BinaryANNTreeSearchBase(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBinaryANNTreeSearchBase.hxx"
#endif

#endif // end #ifndef itkBinaryANNTreeSearchBase_h
