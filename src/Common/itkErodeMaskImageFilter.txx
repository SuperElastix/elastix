#ifndef _itkErodeMaskImageFilter_txx
#define _itkErodeMaskImageFilter_txx

#include "itkErodeMaskImageFilter.h"


namespace itk
{


  /** 
   * ************* Constructor *******************
   */

  template< class TImage >
  ErodeMaskImageFilter< TImage >
  ::ErodeMaskImageFilter()
  {
    for (unsigned int i = 0; i < InputImageDimension; ++i)
    {
      /** Instantiate erosion filter */
      ErodeFilterPointer erosion = ErodeFilterType::New();

      /** Store pointer in array */
      this->m_ErodeFilterArray[i] = erosion;
      
      /** Set up filter */
      erosion->SetForegroundValue( NumericTraits<InputPixelType>::One );
      erosion->SetBackgroundValue( NumericTraits<InputPixelType>::Zero );

      /** Connect the pipeline. */
			if ( i > 0 )
      {
         erosion->SetInput( this->m_ErodeFilterArray[ i - 1 ]->GetOutput() );			
			}
    }

    this->m_IsMovingMask = false;
    this->m_ResolutionLevel = 0;

    ScheduleType defaultSchedule(1,InputImageDimension);
    defaultSchedule.Fill( NumericTraits< unsigned int >::One );
    this->m_Schedule = defaultSchedule;

  } // end Constructor


  /** 
   * ************* GenerateData *******************
   */

  template< class TImage >
  void
  ErodeMaskImageFilter< TImage >
  ::GenerateData()
  {
    
    OutputImagePointer output = this->GetOutput();
    unsigned int level = this->GetResolutionLevel();

    for (unsigned int i = 0; i < InputImageDimension; ++i)
    {
      /** Declare radius-array and structuring element. */
			RadiusType								radiusarray;
			StructuringElementType		S_ball;
      unsigned long radius;
      unsigned int schedule;

      /** Create the radius array */
			radiusarray.Fill( 0 );
			schedule = this->GetSchedule()[ level ][ i ];
      if ( ! this->GetIsMovingMask() )
      {
			  radius = static_cast<unsigned long>( schedule + 1 );
      }
      else
      {
        radius = static_cast<unsigned long>( 2 * schedule + 1 );
      }
			radiusarray.SetElement( i, radius );
      
      /** Create the structuring element and set it into the erosion filter. */
			S_ball.SetRadius( radiusarray );
			S_ball.CreateStructuringElement();
			this->m_ErodeFilterArray[ i ]->SetKernel( S_ball );
    }

    /** Set the input into the first erosion filter */
    this->m_ErodeFilterArray[0]->SetInput(this->GetInput());
  
    // graft this filter's output to the mini-pipeline.  this sets up
    // the mini-pipeline to write to this filter's output and copies
    // region ivars and meta-data
    this->m_ErodeFilterArray[ InputImageDimension - 1 ]->GraftOutput(output);
  
    // execute the mini-pipeline
    this->m_ErodeFilterArray[ InputImageDimension - 1 ]->Update();
  
    // graft the output of the mini-pipeline back onto the filter's output.
    // this copies back the region ivars and meta-dataig
    this->GraftOutput(
      this->m_ErodeFilterArray[ InputImageDimension - 1 ]->GetOutput() );

  } // end GenerateData
  
} // end namespace itk

#endif
