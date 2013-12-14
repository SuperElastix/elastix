/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxVarianceOverLastDimensionMetric_HXX__
#define __elxVarianceOverLastDimensionMetric_HXX__

#include "elxVarianceOverLastDimensionMetric.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
VarianceOverLastDimensionMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of VarianceOverLastDimensionMetric metric took: "
         << static_cast< long >( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

}   // end Initialize


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
VarianceOverLastDimensionMetric< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Get and set the random sampling in the last dimension. */
  bool useRandomSampling = false;
  this->GetConfiguration()->ReadParameter( useRandomSampling,
    "SampleLastDimensionRandomly", this->GetComponentLabel(), level, 0 );
  this->SetSampleLastDimensionRandomly( useRandomSampling );

  /** Get and set if we want to subtract the mean from the derivative. */
  bool subtractMean = false;
  this->GetConfiguration()->ReadParameter( subtractMean,
    "SubtractMean", this->GetComponentLabel(), 0, 0 );
  this->SetSubtractMean( subtractMean );

  /** Get and set the number of random samples for the last dimension. */
  int numSamplesLastDimension = 10;
  this->GetConfiguration()->ReadParameter( numSamplesLastDimension,
    "NumSamplesLastDimension", this->GetComponentLabel(), level, 0 );
  this->SetNumSamplesLastDimension( numSamplesLastDimension );

  /** Get and set the number of additional samples sampled at the fixed timepoint.  */
  unsigned int numAdditionalSamplesFixed = 0;
  this->GetConfiguration()->ReadParameter( numAdditionalSamplesFixed,
    "NumAdditionalSamplesFixed", this->GetComponentLabel(), level, 0 );
  this->SetNumAdditionalSamplesFixed( numAdditionalSamplesFixed );

  /** Get and set the fixed timepoint number. */
  unsigned int reducedDimensionIndex = 0;
  this->GetConfiguration()->ReadParameter(
    reducedDimensionIndex, "ReducedDimensionIndex",
    this->GetComponentLabel(), 0, 0 );
  this->SetReducedDimensionIndex( reducedDimensionIndex );

  /** Set moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales( false );
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  bool                            usescales = true;
  for( unsigned int i = 0; i < MovingImageDimension; ++i )
  {
    usescales = usescales && this->GetConfiguration()->ReadParameter(
      movingImageDerivativeScales[ i ], "MovingImageDerivativeScales",
      this->GetComponentLabel(), i, -1, true );
  }
  if( usescales )
  {
    this->SetUseMovingImageDerivativeScales( true );
    this->SetMovingImageDerivativeScales( movingImageDerivativeScales );
    elxout << "Multiplying moving image derivatives by: "
           << movingImageDerivativeScales << std::endl;
  }

  /** Check if this transform is a B-spline transform. */
  CombinationTransformType * testPtr1
    = dynamic_cast< CombinationTransformType * >( this->GetElastix()->GetElxTransformBase() );
  if( testPtr1 )
  {
    /** Check for B-spline transform. */
    BSplineTransformBaseType * testPtr2 = dynamic_cast< BSplineTransformBaseType * >(
      testPtr1->GetCurrentTransform() );
    if( testPtr2 )
    {
      this->SetGridSize( testPtr2->GetGridRegion().GetSize() );
    }
    else
    {
      /** Check for stack transform. */
      StackTransformType * testPtr3 = dynamic_cast< StackTransformType * >(
        testPtr1->GetCurrentTransform() );
      if( testPtr3 )
      {
        /** Set itk member variable. */
        this->SetTransformIsStackTransform( true );

        if( testPtr3->GetNumberOfSubTransforms() > 0 )
        {
          /** Check if subtransform is a B-spline transform. */
          ReducedDimensionBSplineTransformBaseType * testPtr4 = dynamic_cast< ReducedDimensionBSplineTransformBaseType * >(
            testPtr3->GetSubTransform( 0 ).GetPointer() );
          if( testPtr4 )
          {
            FixedImageSizeType gridSize;
            gridSize.Fill( testPtr3->GetNumberOfSubTransforms() );
            this->SetGridSize( gridSize );
          }
        }
      }
    }
  }

}   // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxVarianceOverLastDimensionMetric_HXX__
