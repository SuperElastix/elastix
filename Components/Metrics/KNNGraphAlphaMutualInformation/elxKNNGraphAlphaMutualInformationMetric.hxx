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
#ifndef elxKNNGraphAlphaMutualInformationMetric_hxx
#define elxKNNGraphAlphaMutualInformationMetric_hxx

#include "elxKNNGraphAlphaMutualInformationMetric.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkTimeProbe.h"
#include <itkDeref.h>
#include <string>

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
KNNGraphAlphaMutualInformationMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of KNNGraphAlphaMutualInformation metric took: "
                                 << static_cast<std::int64_t>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
KNNGraphAlphaMutualInformationMetric<TElastix>::BeforeRegistration()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Get and set alpha, from alpha - MI. */
  double alpha = 0.5;
  configuration.ReadParameter(alpha, "Alpha", 0);
  this->SetAlpha(alpha);

  /** Get the small number that avoids division by that number. */
  double smallNumber = 1e-5;
  configuration.ReadParameter(smallNumber, "AvoidDivisionBy", 0, true);
  this->SetAvoidDivisionBy(smallNumber);

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
KNNGraphAlphaMutualInformationMetric<TElastix>::BeforeEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Get the current resolution level. */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Get the parameters for the KNN binary tree. */

  /** Get the tree type. */
  std::string treeType = "KDTree";
  configuration.ReadParameter(treeType, "TreeType", 0);
  configuration.ReadParameter(treeType, "TreeType", level, true);

  bool silentBS = false;
  bool silentSplit = false;
  bool silentShrink = false;
  if (treeType == "KDTree")
  {
    silentShrink = true;
  }
  else if (treeType == "BruteForceTree")
  {
    silentBS = true;
    silentSplit = true;
    silentShrink = true;
  }

  /** Get the bucket size. */
  unsigned int bucketSize = 50;
  configuration.ReadParameter(bucketSize, "BucketSize", 0, silentBS);
  configuration.ReadParameter(bucketSize, "BucketSize", level, true);

  /** Get the splitting rule for all trees. */
  std::string splittingRule = "ANN_KD_SL_MIDPT";
  bool        returnValue = configuration.ReadParameter(splittingRule, "SplittingRule", 0, silentSplit);
  configuration.ReadParameter(splittingRule, "SplittingRule", level, true);

  /** Get the splitting rule for the fixed tree. */
  std::string fixedSplittingRule = splittingRule;
  configuration.ReadParameter(fixedSplittingRule, "FixedSplittingRule", 0, silentSplit | !returnValue);
  configuration.ReadParameter(fixedSplittingRule, "FixedSplittingRule", level, true);

  /** Get the splitting rule for the moving tree. */
  std::string movingSplittingRule = splittingRule;
  configuration.ReadParameter(movingSplittingRule, "MovingSplittingRule", 0, silentSplit | !returnValue);
  configuration.ReadParameter(movingSplittingRule, "MovingSplittingRule", level, true);

  /** Get the splitting rule for the joint tree. */
  std::string jointSplittingRule = splittingRule;
  configuration.ReadParameter(jointSplittingRule, "JointSplittingRule", 0, silentSplit | !returnValue);
  configuration.ReadParameter(jointSplittingRule, "JointSplittingRule", level, true);

  /** Get the shrinking rule for all trees. */
  std::string shrinkingRule = "ANN_BD_SIMPLE";
  returnValue = configuration.ReadParameter(shrinkingRule, "ShrinkingRule", 0, silentShrink);
  configuration.ReadParameter(shrinkingRule, "ShrinkingRule", level, true);

  /** Get the shrinking rule for the fixed tree. */
  std::string fixedShrinkingRule = shrinkingRule;
  configuration.ReadParameter(fixedShrinkingRule, "FixedShrinkingRule", 0, silentShrink | !returnValue);
  configuration.ReadParameter(fixedShrinkingRule, "FixedShrinkingRule", level, true);

  /** Get the shrinking rule for the moving tree. */
  std::string movingShrinkingRule = shrinkingRule;
  configuration.ReadParameter(movingShrinkingRule, "MovingShrinkingRule", 0, silentShrink | !returnValue);
  configuration.ReadParameter(movingShrinkingRule, "MovingShrinkingRule", level, true);

  /** Get the shrinking rule for the joint tree. */
  std::string jointShrinkingRule = shrinkingRule;
  configuration.ReadParameter(jointShrinkingRule, "JointShrinkingRule", 0, silentShrink | !returnValue);
  configuration.ReadParameter(jointShrinkingRule, "JointShrinkingRule", level, true);

  /** Set the tree. */
  if (treeType == "KDTree")
  {
    this->SetANNkDTree(bucketSize, fixedSplittingRule, movingSplittingRule, jointSplittingRule);
  }
  else if (treeType == "BDTree")
  {
    this->SetANNbdTree(bucketSize,
                       fixedSplittingRule,
                       movingSplittingRule,
                       jointSplittingRule,
                       fixedShrinkingRule,
                       movingShrinkingRule,
                       jointShrinkingRule);
  }
  else if (treeType == "BruteForceTree")
  {
    this->SetANNBruteForceTree();
  }
  else
  {
    itkExceptionMacro("ERROR: there is no tree type \"" << treeType << "\" implemented.");
  }

  /** Get the parameters for the search tree. */

  /** Get the tree search type. */
  std::string treeSearchType = "Standard";
  configuration.ReadParameter(treeSearchType, "TreeSearchType", 0);
  configuration.ReadParameter(treeSearchType, "TreeSearchType", level, true);

  bool silentSR = true;
  if (treeSearchType == "FixedRadius")
  {
    silentSR = false;
  }

  /** Get the k nearest neighbours. */
  unsigned int kNearestNeighbours = 20;
  configuration.ReadParameter(kNearestNeighbours, "KNearestNeighbours", 0);
  configuration.ReadParameter(kNearestNeighbours, "KNearestNeighbours", level, true);

  /** Get the error bound. */
  double errorBound = 0.0;
  configuration.ReadParameter(errorBound, "ErrorBound", 0);
  configuration.ReadParameter(errorBound, "ErrorBound", level, true);

  /** Get the squared search radius. */
  double squaredSearchRadius = 0.0;
  configuration.ReadParameter(squaredSearchRadius, "SquaredSearchRadius", 0, silentSR);
  configuration.ReadParameter(squaredSearchRadius, "SquaredSearchRadius", level, true);

  /** Set the tree searcher. */
  if (treeSearchType == "Standard")
  {
    this->SetANNStandardTreeSearch(kNearestNeighbours, errorBound);
  }
  else if (treeSearchType == "FixedRadius")
  {
    this->SetANNFixedRadiusTreeSearch(kNearestNeighbours, errorBound, squaredSearchRadius);
  }
  else if (treeSearchType == "Priority")
  {
    this->SetANNPriorityTreeSearch(kNearestNeighbours, errorBound);
  }
  else
  {
    itkExceptionMacro("ERROR: there is no tree searcher type \"" << treeSearchType << "\" implemented.");
  }

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxKNNGraphAlphaMutualInformationMetric_hxx
