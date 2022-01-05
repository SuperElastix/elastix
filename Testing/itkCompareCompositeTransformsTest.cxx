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
// ITK transforms
#include "itkAffineTransform.h"
#include "itkTranslationTransform.h"
#include "itkBSplineTransform.h"
#include "itkCompositeTransform.h"

// elastix GPU transforms
#include "itkAdvancedMatrixOffsetTransformBase.h"
#include "itkAdvancedTranslationTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"

#include <iomanip>
#include <type_traits> // For std::is_base.

namespace
{
const unsigned int Dimension = 3;
typedef float      ScalarType;

template <typename TDerivedTransform>
bool
IsDerivedTransformOfType(const itk::SmartPointer<itk::Transform<ScalarType, Dimension, Dimension>> & ptr)
{
  static_assert(std::is_base_of<itk::Transform<ScalarType, Dimension, Dimension>, TDerivedTransform>::value,
                "TDerivedTransform must be derived from itk::Transform!");
  return dynamic_cast<const TDerivedTransform *>(ptr.GetPointer()) != nullptr;
}
} // namespace

//-------------------------------------------------------------------------------------
int
main()
{
  // ITK transform typedefs
  // typedef itk::Transform< ScalarType, Dimension, Dimension > TransformType;
  using AffineTransformType = itk::AffineTransform<ScalarType, Dimension>;
  using TranslationTransformType = itk::TranslationTransform<ScalarType, Dimension>;
  using BSplineTransformType = itk::BSplineTransform<ScalarType, Dimension, 3>;
  using CompositeTransformType = itk::CompositeTransform<ScalarType, Dimension>;

  // elastix advanced transform typedefs
  using AdvancedCombinationTransformType = itk::AdvancedCombinationTransform<ScalarType, Dimension>;
  // typedef itk::AdvancedTransform< ScalarType, Dimension, Dimension >
  // AdvancedTransformType;
  using AdvancedAffineTransformType = itk::AdvancedMatrixOffsetTransformBase<ScalarType, Dimension, Dimension>;
  using AdvancedTranslationTransformType = itk::AdvancedTranslationTransform<ScalarType, Dimension>;
  using AdvancedBSplineTransformType = itk::AdvancedBSplineDeformableTransform<ScalarType, Dimension, 3>;

  // Define ITK transforms
  const auto affine = AffineTransformType::New();
  const auto translation = TranslationTransformType::New();
  const auto bspline = BSplineTransformType::New();

  // Define ITK composite transform and test it
  const auto composite = CompositeTransformType::New();

  if (composite->GetNumberOfTransforms() != 0)
  {
    std::cerr << "Error in getting number of transforms from itk::CompositeTransform." << std::endl;
    return EXIT_FAILURE;
  }

  composite->AddTransform(affine);
  composite->AddTransform(translation);
  composite->AddTransform(bspline);

  if (composite->GetNumberOfTransforms() != 3)
  {
    std::cerr << "Error in getting number of transforms from itk::CompositeTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Define elastix advanced composite transforms
  const auto advancedAffine = AdvancedAffineTransformType::New();
  const auto advancedTranslation = AdvancedTranslationTransformType::New();
  const auto advancedBSpline = AdvancedBSplineTransformType::New();

  // Define elastix advanced composite transform and test it
  const auto advancedComposite = AdvancedCombinationTransformType::New();
  if (advancedComposite->GetNumberOfTransforms() != 0)
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  advancedComposite->SetCurrentTransform(advancedAffine);

  if (advancedComposite->GetNumberOfTransforms() != 1)
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Define the next one
  const auto advancedCompositeTranslation = AdvancedCombinationTransformType::New();
  advancedCompositeTranslation->SetCurrentTransform(advancedTranslation);
  advancedComposite->SetInitialTransform(advancedCompositeTranslation);

  if (advancedCompositeTranslation->GetNumberOfTransforms() != 1)
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (advancedComposite->GetNumberOfTransforms() != 2)
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Define the next one
  const auto advancedCompositeBSpline = AdvancedCombinationTransformType::New();
  advancedCompositeBSpline->SetCurrentTransform(advancedBSpline);
  advancedCompositeTranslation->SetInitialTransform(advancedCompositeBSpline);

  if (advancedCompositeBSpline->GetNumberOfTransforms() != 1)
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (advancedCompositeTranslation->GetNumberOfTransforms() != 2)
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (advancedComposite->GetNumberOfTransforms() != 3)
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Checks for GetNthTransform for itk::CompositeTransform
  if (!IsDerivedTransformOfType<AffineTransformType>(composite->GetNthTransform(0)))
  {
    std::cerr << "Error expecting AffineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (!IsDerivedTransformOfType<TranslationTransformType>(composite->GetNthTransform(1)))
  {
    std::cerr << "Error expecting TranslationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (!IsDerivedTransformOfType<BSplineTransformType>(composite->GetNthTransform(2)))
  {
    std::cerr << "Error expecting BSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Checks for GetNthTransform for itk::AdvancedCombinationTransform
  if (!IsDerivedTransformOfType<AdvancedAffineTransformType>(advancedComposite->GetNthTransform(0)))
  {
    std::cerr << "Error expecting AdvancedAffineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (!IsDerivedTransformOfType<AdvancedTranslationTransformType>(advancedComposite->GetNthTransform(1)))
  {
    std::cerr << "Error expecting AdvancedTranslationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (!IsDerivedTransformOfType<AdvancedBSplineTransformType>(advancedComposite->GetNthTransform(2)))
  {
    std::cerr << "Error expecting AdvancedBSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (!IsDerivedTransformOfType<AdvancedTranslationTransformType>(advancedCompositeTranslation->GetNthTransform(0)))
  {
    std::cerr << "Error expecting AdvancedTranslationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (!IsDerivedTransformOfType<AdvancedBSplineTransformType>(advancedCompositeTranslation->GetNthTransform(1)))
  {
    std::cerr << "Error expecting AdvancedBSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if (!IsDerivedTransformOfType<AdvancedBSplineTransformType>(advancedCompositeBSpline->GetNthTransform(0)))
  {
    std::cerr << "Error expecting AdvancedBSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  /** Return a value. */
  return EXIT_SUCCESS;
} // end main
