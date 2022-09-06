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
#ifndef itkANNkDTree_h
#define itkANNkDTree_h

#include "itkBinaryANNTreeBase.h"

namespace itk
{

/**
 * \class ANNkDTree
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT ANNkDTree : public BinaryANNTreeBase<TListSample>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ANNkDTree);

  /** Standard itk. */
  using Self = ANNkDTree;
  using Superclass = BinaryANNTreeBase<TListSample>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK type info. */
  itkTypeMacro(ANNkDTree, BinaryANNTreeBase);

  /** Typedef's from Superclass. */
  using typename Superclass::SampleType;
  using typename Superclass::MeasurementVectorType;
  using typename Superclass::MeasurementVectorSizeType;
  using typename Superclass::TotalAbsoluteFrequencyType;

  /** Typedef's. */
  using ANNPointSetType = ANNpointSet;
  using ANNkDTreeType = ANNkd_tree;
  using SplittingRuleType = ANNsplitRule;
  using BucketSizeType = unsigned int;

  /** Set and get the bucket size: the number of points in a region/bucket. */
  itkSetMacro(BucketSize, BucketSizeType);
  itkGetConstMacro(BucketSize, BucketSizeType);

  /** Set and get the splitting rule: it defines how the space is divided. */
  itkSetMacro(SplittingRule, SplittingRuleType);
  itkGetConstMacro(SplittingRule, SplittingRuleType);
  void
  SetSplittingRule(const std::string & rule);

  std::string
  GetSplittingRule();

  /** Set the maximum number of points that are to be visited. */
  // void SetMaximumNumberOfPointsToVisit( unsigned int num )
  //{
  //  annMaxPtsVisit( static_cast<int>( num ) );
  //}

  /** Generate the tree. */
  void
  GenerateTree() override;

  /** Get the ANN tree. */
  ANNPointSetType *
  GetANNTree() const override
  {
    return this->m_ANNTree;
  }


protected:
  /** Constructor. */
  ANNkDTree();

  /** Destructor. */
  ~ANNkDTree() override;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variables. */
  ANNkDTreeType *   m_ANNTree;
  SplittingRuleType m_SplittingRule;
  BucketSizeType    m_BucketSize;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkANNkDTree.hxx"
#endif

#endif // end #ifndef itkANNkDTree_h
