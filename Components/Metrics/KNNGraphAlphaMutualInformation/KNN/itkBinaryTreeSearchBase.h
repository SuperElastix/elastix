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
#ifndef itkBinaryTreeSearchBase_h
#define itkBinaryTreeSearchBase_h

#include "itkObject.h"
#include "itkArray.h"

#include "itkBinaryTreeBase.h"

namespace itk
{

/**
 * \class BinaryTreeSearchBase
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT BinaryTreeSearchBase : public Object
{
public:
  /** Standard itk. */
  typedef BinaryTreeSearchBase     Self;
  typedef Object                   Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** ITK type info. */
  itkTypeMacro(BinaryTreeSearchBase, Object);

  /** Typedef's. */
  typedef TListSample                                    ListSampleType;
  typedef BinaryTreeBase<ListSampleType>                 BinaryTreeType;
  typedef typename BinaryTreeType::Pointer               BinaryTreePointer;
  typedef typename BinaryTreeType::MeasurementVectorType MeasurementVectorType;
  typedef Array<int>                                     IndexArrayType;
  typedef Array<double>                                  DistanceArrayType;

  /** Set and get the binary tree. */
  virtual void
  SetBinaryTree(BinaryTreeType * tree);

  const BinaryTreeType *
  GetBinaryTree(void) const;

  /** Set and get the number of nearest neighbours k. */
  itkSetMacro(KNearestNeighbors, unsigned int);
  itkGetConstMacro(KNearestNeighbors, unsigned int);

  /** Search the nearest neighbours of a query point qp. */
  virtual void
  Search(const MeasurementVectorType & qp, IndexArrayType & ind, DistanceArrayType & dists) = 0;

protected:
  BinaryTreeSearchBase();
  ~BinaryTreeSearchBase() override;

  /** Member variables. */
  BinaryTreePointer m_BinaryTree;
  unsigned int      m_KNearestNeighbors;
  unsigned int      m_DataDimension;

private:
  BinaryTreeSearchBase(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBinaryTreeSearchBase.hxx"
#endif

#endif // end #ifndef itkBinaryTreeSearchBase_h
