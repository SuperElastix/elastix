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
#ifndef itkANNbdTree_h
#define itkANNbdTree_h

#include "itkANNkDTree.h"

namespace itk
{

/**
 * \class ANNbdTree
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT ANNbdTree : public ANNkDTree<TListSample>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ANNbdTree);

  /** Standard itk. */
  using Self = ANNbdTree;
  using Superclass = ANNkDTree<TListSample>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK type info. */
  itkTypeMacro(ANNbdTree, ANNkDTree);

  /** Typedef's from Superclass. */
  using typename Superclass::SampleType;
  using typename Superclass::MeasurementVectorType;
  using typename Superclass::MeasurementVectorSizeType;
  using typename Superclass::TotalAbsoluteFrequencyType;
  using typename Superclass::ANNPointSetType;
  using typename Superclass::ANNkDTreeType;
  using typename Superclass::SplittingRuleType;
  using typename Superclass::BucketSizeType;

  using ShrinkingRuleType = ANNshrinkRule;

  /** Set and get the shrinking rule: it defines ... */
  itkSetMacro(ShrinkingRule, ShrinkingRuleType);
  itkGetConstMacro(ShrinkingRule, ShrinkingRuleType);
  void
  SetShrinkingRule(const std::string & rule);

  std::string
  GetShrinkingRule();

  /** Generate the tree. */
  void
  GenerateTree() override;

protected:
  /** Constructor. */
  ANNbdTree();

  /** Destructor. */
  ~ANNbdTree() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variables. */
  ShrinkingRuleType m_ShrinkingRule;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkANNbdTree.hxx"
#endif

#endif // end #ifndef itkANNbdTree_h
