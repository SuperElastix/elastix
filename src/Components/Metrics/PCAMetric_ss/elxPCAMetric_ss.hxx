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
#ifndef __elxPCAMetric_ss_HXX__
#define __elxPCAMetric_ss_HXX__

#include "elxPCAMetric_ss.h"
#include "itkTimeProbe.h"


namespace elastix
{

  /**
   * ******************* Initialize ***********************
   */

  template <class TElastix>
    void PCAMetric_ss<TElastix>
      ::Initialize(void) throw (itk::ExceptionObject)
  {
    itk::TimeProbe timer;
    timer.Start();
    this->Superclass1::Initialize();
    timer.Stop();
    elxout << "Initialization of PCAMetric_ss metric took: "
      << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

  } // end Initialize()


  /**
   * ***************** BeforeEachResolution ***********************
   */

  template <class TElastix>
    void PCAMetric_ss<TElastix>
    ::BeforeEachResolution(void)
  {
    /** Get the current resolution level. */
    unsigned int level =
      ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    unsigned int NumEigenValues = 6;
    this->GetConfiguration()->ReadParameter( NumEigenValues, "NumEigenValues",
        this->GetComponentLabel(), level, 0);
    this->SetNumEigenValues( NumEigenValues );

    unsigned int NumSingleSubjects = 1;
    this->GetConfiguration()->ReadParameter( NumSingleSubjects, "NumSingleSubjects",
        this->GetComponentLabel(), level, 0);
    this->SetNumSingleSubjects( NumSingleSubjects );

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
    elxout << "end BeforeEachResolution" << std::endl;

  } // end BeforeEachResolution

} // end namespace elastix


#endif // end #ifndef __elxPCAMetric_ss_HXX__

