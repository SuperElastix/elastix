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
using namespace itk;

  /**
   * ******************* Initialize ***********************
   */

  template <class TElastix>
    void VarianceOverLastDimensionMetric<TElastix>
    ::Initialize(void) throw (ExceptionObject)
  {
    TimerPointer timer = TimerType::New();
    timer->StartTimer();
    this->Superclass1::Initialize();
    timer->StopTimer();
    elxout << "Initialization of VarianceOverLastDimensionMetric metric took: "
      << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

  } // end Initialize


  /**
   * ***************** BeforeEachResolution ***********************
   */

  template <class TElastix>
    void VarianceOverLastDimensionMetric<TElastix>
    ::BeforeEachResolution(void)
  {
    /** Get the current resolution level. */
    unsigned int level = 
      ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Get and set the random sampling in the last dimension. */
    bool useRandomSampling = false;
    this->GetConfiguration()->ReadParameter( useRandomSampling,
      "SampleLastDimensionRandomly", this->GetComponentLabel(), level, 0 );
    this->SetSampleLastDimensionRandomly( useRandomSampling );

    /** Get and set the number of random samples for the last dimension. */
    int numSamplesLastDimension = 10;
    this->GetConfiguration()->ReadParameter( numSamplesLastDimension,
      "NumSamplesLastDimension", this->GetComponentLabel(), level, 0 );
    this->SetNumSamplesLastDimension( numSamplesLastDimension );

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
    
  } // end BeforeEachResolution

} // end namespace elastix


#endif // end #ifndef __elxVarianceOverLastDimensionMetric_HXX__

