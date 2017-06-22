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

//-------------------------------------------------------------------------------------
int
main( int argc, char * argv[] )
{
  const unsigned int Dimension = 3;
  typedef float ScalarType;

  // ITK transform typedefs
  //typedef itk::Transform< ScalarType, Dimension, Dimension > TransformType;
  typedef itk::AffineTransform< ScalarType, Dimension >      AffineTransformType;
  typedef itk::TranslationTransform< ScalarType, Dimension > TranslationTransformType;
  typedef itk::BSplineTransform< ScalarType, Dimension, 3 >  BSplineTransformType;
  typedef itk::CompositeTransform< ScalarType, Dimension >   CompositeTransformType;

  // elastix advanced transform typedefs
  typedef itk::AdvancedCombinationTransform< ScalarType, Dimension >
    AdvancedCombinationTransformType;
  //typedef itk::AdvancedTransform< ScalarType, Dimension, Dimension >
  //AdvancedTransformType;
  typedef itk::AdvancedMatrixOffsetTransformBase< ScalarType, Dimension, Dimension >
    AdvancedAffineTransformType;
  typedef itk::AdvancedTranslationTransform< ScalarType, Dimension >
    AdvancedTranslationTransformType;
  typedef itk::AdvancedBSplineDeformableTransform< ScalarType, Dimension, 3 >
    AdvancedBSplineTransformType;

  // Define ITK transforms
  AffineTransformType::Pointer      affine      = AffineTransformType::New();
  TranslationTransformType::Pointer translation = TranslationTransformType::New();
  BSplineTransformType::Pointer     bspline     = BSplineTransformType::New();

  // Define ITK composite transform and test it
  CompositeTransformType::Pointer composite = CompositeTransformType::New();

  if( composite->GetNumberOfTransforms() != 0 )
  {
    std::cerr << "Error in getting number of transforms from itk::CompositeTransform." << std::endl;
    return EXIT_FAILURE;
  }

  composite->AddTransform( affine );
  composite->AddTransform( translation );
  composite->AddTransform( bspline );

  if( composite->GetNumberOfTransforms() != 3 )
  {
    std::cerr << "Error in getting number of transforms from itk::CompositeTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Define elastix advanced composite transforms
  AdvancedAffineTransformType::Pointer advancedAffine
    = AdvancedAffineTransformType::New();
  AdvancedTranslationTransformType::Pointer advancedTranslation
    = AdvancedTranslationTransformType::New();
  AdvancedBSplineTransformType::Pointer advancedBSpline
    = AdvancedBSplineTransformType::New();

  // Define elastix advanced composite transform and test it
  AdvancedCombinationTransformType::Pointer advancedComposite
    = AdvancedCombinationTransformType::New();
  if( advancedComposite->GetNumberOfTransforms() != 0 )
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  advancedComposite->SetCurrentTransform( advancedAffine );

  if( advancedComposite->GetNumberOfTransforms() != 1 )
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Define the next one
  AdvancedCombinationTransformType::Pointer advancedCompositeTranslation
    = AdvancedCombinationTransformType::New();
  advancedCompositeTranslation->SetCurrentTransform( advancedTranslation );
  advancedComposite->SetInitialTransform( advancedCompositeTranslation );

  if( advancedCompositeTranslation->GetNumberOfTransforms() != 1 )
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if( advancedComposite->GetNumberOfTransforms() != 2 )
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Define the next one
  AdvancedCombinationTransformType::Pointer advancedCompositeBSpline
    = AdvancedCombinationTransformType::New();
  advancedCompositeBSpline->SetCurrentTransform( advancedBSpline );
  advancedCompositeTranslation->SetInitialTransform( advancedCompositeBSpline );

  if( advancedCompositeBSpline->GetNumberOfTransforms() != 1 )
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if( advancedCompositeTranslation->GetNumberOfTransforms() != 2 )
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  if( advancedComposite->GetNumberOfTransforms() != 3 )
  {
    std::cerr << "Error in getting number of transforms from itk::AdvancedCombinationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Checks for GetNthTransform for itk::CompositeTransform
  const AffineTransformType * isAffine
    = dynamic_cast< const AffineTransformType * >
    ( composite->GetNthTransform( 0 ).GetPointer() );
  if( !isAffine )
  {
    std::cerr << "Error expecting AffineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  const TranslationTransformType * isTranslation
    = dynamic_cast< const TranslationTransformType * >
    ( composite->GetNthTransform( 1 ).GetPointer() );
  if( !isTranslation )
  {
    std::cerr << "Error expecting TranslationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  const BSplineTransformType * isBSpline
    = dynamic_cast< const BSplineTransformType * >
    ( composite->GetNthTransform( 2 ).GetPointer() );
  if( !isBSpline )
  {
    std::cerr << "Error expecting BSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  // Checks for GetNthTransform for itk::AdvancedCombinationTransform
  const AdvancedAffineTransformType * isAdvancedAffine0
    = dynamic_cast< const AdvancedAffineTransformType * >
    ( advancedComposite->GetNthTransform( 0 ).GetPointer() );
  if( !isAdvancedAffine0 )
  {
    std::cerr << "Error expecting AdvancedAffineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  const AdvancedTranslationTransformType * isAdvancedTranslation0
    = dynamic_cast< const AdvancedTranslationTransformType * >
    ( advancedComposite->GetNthTransform( 1 ).GetPointer() );
  if( !isAdvancedTranslation0 )
  {
    std::cerr << "Error expecting AdvancedTranslationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  const AdvancedBSplineTransformType * isAdvancedBSpline0
    = dynamic_cast< const AdvancedBSplineTransformType * >
    ( advancedComposite->GetNthTransform( 2 ).GetPointer() );
  if( !isAdvancedBSpline0 )
  {
    std::cerr << "Error expecting AdvancedBSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  const AdvancedTranslationTransformType * isAdvancedTranslation1
    = dynamic_cast< const AdvancedTranslationTransformType * >
    ( advancedCompositeTranslation->GetNthTransform( 0 ).GetPointer() );
  if( !isAdvancedTranslation1 )
  {
    std::cerr << "Error expecting AdvancedTranslationTransform." << std::endl;
    return EXIT_FAILURE;
  }

  const AdvancedBSplineTransformType * isAdvancedBSpline1
    = dynamic_cast< const AdvancedBSplineTransformType * >
    ( advancedCompositeTranslation->GetNthTransform( 1 ).GetPointer() );
  if( !isAdvancedBSpline1 )
  {
    std::cerr << "Error expecting AdvancedBSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  const AdvancedBSplineTransformType * isAdvancedBSpline2
    = dynamic_cast< const AdvancedBSplineTransformType * >
    ( advancedCompositeBSpline->GetNthTransform( 0 ).GetPointer() );
  if( !isAdvancedBSpline2 )
  {
    std::cerr << "Error expecting AdvancedBSplineTransform." << std::endl;
    return EXIT_FAILURE;
  }

  /** Return a value. */
  return EXIT_SUCCESS;
} // end main
