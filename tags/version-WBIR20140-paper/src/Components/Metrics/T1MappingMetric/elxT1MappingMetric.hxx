/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxT1MappingMetric_HXX__
#define __elxT1MappingMetricc_HXX__

#include <vector>
#include "elxT1MappingMetric.h"


namespace elastix
{

  /**
   * ******************* Initialize ***********************
   */

  template <class TElastix>
    void T1MappingMetric<TElastix>
      ::Initialize(void) throw (itk::ExceptionObject)
  {
        this->m_iterationCounter = 0;
        //this->m_Simage.set_size(1000,25);
        //this->m_Simage.fill(0.0);
        TimerPointer timer = TimerType::New();
        timer->StartTimer();
        this->Superclass1::Initialize();
        timer->StopTimer();
        elxout << "Initialization of T1Mapping metric took: "
               << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

  } // end Initialize


  /**
   * ***************** BeforeEachResolution ***********************
   */

  template <class TElastix>
    void T1MappingMetric<TElastix>
    ::BeforeEachResolution(void)
  {
        std::cout << "BeforeEachResolution" << std::endl;
    /** Get the current resolution level. */
    unsigned int level =
      ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    const unsigned int n = 100;
    std::vector< double > triggerTimes;
    triggerTimes.resize(n);
   std::fill(triggerTimes.begin(),triggerTimes.end(), 0.0);
    for( unsigned int i = 0; i < n; ++i)
    {
        this->GetConfiguration()->ReadParameter( triggerTimes[ i ],
        "TriggerTimes", i, 0 );
    }
    this->SetTriggerTimes( triggerTimes );

    /** Get and set if we want to subtract the mean from the derivative. */
    bool subtractMean = false;
    this->GetConfiguration()->ReadParameter( subtractMean,
      "SubtractMean", this->GetComponentLabel(), 0, 0 );
    this->SetSubtractMean( subtractMean );

    unsigned int nrofit = 10;
    this->GetConfiguration()->ReadParameter( nrofit,
      "NumberOfIterationsForLM", this->GetComponentLabel(), 0, 0 );
    this->SetNumberOfIterationsForLM( nrofit );

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
    bool usescales = true;
    for ( unsigned int i = 0; i < MovingImageDimension; ++i )
    {
      usescales = usescales && this->GetConfiguration()->ReadParameter(
        movingImageDerivativeScales[ i ], "MovingImageDerivativeScales",
        this->GetComponentLabel(), i, -1, true );
    }
    if ( usescales )
    {
      this->SetUseMovingImageDerivativeScales( true );
      this->SetMovingImageDerivativeScales( movingImageDerivativeScales );
      elxout << "Multiplying moving image derivatives by: "
        << movingImageDerivativeScales << std::endl;
    }

    /** Check if this transform is a B-spline transform. */
    CombinationTransformType * testPtr1
      = dynamic_cast<CombinationTransformType *>( this->GetElastix()->GetElxTransformBase() );
    if ( testPtr1 )
    {
      /** Check for B-spline transform. */
      BSplineTransformBaseType * testPtr2 = dynamic_cast<BSplineTransformBaseType *>(
        testPtr1->GetCurrentTransform() );
      if ( testPtr2 )
      {
        this->SetGridSize( testPtr2->GetGridRegion().GetSize() );
      }
      else
      {
        /** Check for stack transform. */
        StackTransformType * testPtr3 = dynamic_cast<StackTransformType *>(
          testPtr1->GetCurrentTransform() );
        if ( testPtr3 )
        {
          /** Set itk member variable. */
          this->SetTransformIsStackTransform ( true );

          if ( testPtr3->GetNumberOfSubTransforms() > 0 )
          {
            /** Check if subtransform is a B-spline transform. */
            ReducedDimensionBSplineTransformBaseType * testPtr4 = dynamic_cast<ReducedDimensionBSplineTransformBaseType *>(
              testPtr3->GetSubTransform( 0 ).GetPointer() );
            if ( testPtr4 )
            {
              FixedImageSizeType gridSize;
              gridSize.Fill( testPtr3->GetNumberOfSubTransforms() );
              this->SetGridSize( gridSize );
            }
          }
        }
      }
    }

  } // end BeforeEachResolution

/**
  * ***************** BeforeEachResolution ***********************
*/

template <class TElastix>
void T1MappingMetric<TElastix>
      ::AfterEachIteration(void)
    {
        this->m_iterationCounter += 1;
//         if(this->m_iterationCounter % 20 == 0)
//        {
//            elxout << "recalculate S: number of iterations is: " << this->m_iterationCounter << std::endl;
//        }
    }// end AfterEachIteration

} // end namespace elastix


#endif // end #ifndef __elxT1MappingMetric_HXX__

