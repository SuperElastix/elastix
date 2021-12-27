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
#ifndef itkANNbdTree_hxx
#define itkANNbdTree_hxx

#include "itkANNbdTree.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TListSample>
ANNbdTree<TListSample>::ANNbdTree()
{
  this->m_ShrinkingRule = ANN_BD_SIMPLE;

} // end Constructor()


/**
 * ************************ SetShrinkingRule *************************
 */

template <class TListSample>
void
ANNbdTree<TListSample>::SetShrinkingRule(const std::string & rule)
{
  if (rule == "ANN_BD_NONE")
  {
    this->m_ShrinkingRule = ANN_BD_NONE;
  }
  else if (rule == "ANN_BD_SIMPLE")
  {
    this->m_ShrinkingRule = ANN_BD_SIMPLE;
  }
  else if (rule == "ANN_BD_CENTROID")
  {
    this->m_ShrinkingRule = ANN_BD_CENTROID;
  }
  else if (rule == "ANN_BD_SUGGEST")
  {
    this->m_ShrinkingRule = ANN_BD_SUGGEST;
  }
  else
  {
    itkWarningMacro(<< "WARNING: No such shrinking rule.");
  }

} // end SetShrinkingRule()


/**
 * ************************ GetShrinkingRule *************************
 */

template <class TListSample>
std::string
ANNbdTree<TListSample>::GetShrinkingRule()
{
  switch (this->m_ShrinkingRule)
  {
    case ANN_BD_NONE:
      return "ANN_BD_NONE";
    case ANN_BD_SIMPLE:
      return "ANN_BD_SIMPLE";
    case ANN_BD_CENTROID:
      return "ANN_BD_CENTROID";
    case ANN_BD_SUGGEST:
      return "ANN_BD_SUGGEST";
  }

} // end GetShrinkingRule()


/**
 * ************************ GenerateTree *************************
 */

template <class TListSample>
void
ANNbdTree<TListSample>::GenerateTree()
{
  int dim = static_cast<int>(this->GetDataDimension());
  int nop = static_cast<int>(this->GetActualNumberOfDataPoints());
  int bcs = static_cast<int>(this->m_BucketSize);

  ANNBinaryTreeCreator::DeleteANNkDTree(this->m_ANNTree);

  this->m_ANNTree = ANNBinaryTreeCreator::CreateANNbdTree(
    this->GetSample()->GetInternalContainer(), nop, dim, bcs, this->m_SplittingRule, this->m_ShrinkingRule);

} // end GenerateTree()


/**
 * ************************ PrintSelf *************************
 */

template <class TListSample>
void
ANNbdTree<TListSample>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "ShrinkingRule: " << this->m_ShrinkingRule << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkANNbdTree_hxx
