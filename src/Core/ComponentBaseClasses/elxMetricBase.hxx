#ifndef __elxMetricBase_hxx
#define __elxMetricBase_hxx

#include "elxMetricBase.h"

#include "itkImageSamplerBase.h"
#include "itkImageRandomSampler.h"
#include "itkImageRandomSamplerSparseMask.h"
#include "itkImageFullSampler.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkImageGridSampler.h"

namespace elastix
{
	using namespace itk;

	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MetricBase<TElastix>::MetricBase()
	{
		/** Initialize. */
    this->m_ShowExactMetricValue = 0;
    this->m_ExactMetricSampler = 0;

	} // end Constructor

			
	/**
	 * ******************* BeforeEachResolutionBase ******************
	 */

	template <class TElastix>
		void MetricBase<TElastix>
		::BeforeEachResolutionBase(void)
	{
		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Configure the image sampler. */
    this->ConfigureImageSampler();

    /** Check if the exact metric value, computed on all pixels, should be shown, 
		 * and whether the all pixels should be used during optimisation */

    /** Remove the ExactMetric-column, if it already existed. */
		xl::xout["iteration"].RemoveTargetCell("ExactMetric");
    /** Read the parameter file: Show the exact metric in every iteration? */ 
		std::string showExactMetricValue = "false";
    this->GetConfiguration()->
		  ReadParameter(showExactMetricValue, "ShowExactMetricValue", 0, true);
		this->GetConfiguration()->
			ReadParameter(showExactMetricValue, "ShowExactMetricValue", level);
		if (showExactMetricValue == "true")
		{
      /** Create a new column in the iteration info table */
			xl::xout["iteration"].AddTargetCell("ExactMetric");
			xl::xout["iteration"]["ExactMetric"] << std::showpoint << std::fixed;
      /** Remember that we want to show the exact metric value */
      this->m_ShowExactMetricValue = true;
		}
		else
		{
			this->m_ShowExactMetricValue = false;
		}
  
	} // end BeforeEachResolutionBase


  /**
	 * ******************* AfterEachIterationBase ******************
	 */

	template <class TElastix>
		void MetricBase<TElastix>
		::AfterEachIterationBase(void)
	{ 
		/** Show the metric value computed on all voxels,
		 * if the user wanted it */
		if (this->m_ShowExactMetricValue)
		{
			xl::xout["iteration"]["ExactMetric"] << this->GetExactValue(
				this->GetElastix()->
				GetElxOptimizerBase()->GetAsITKBaseType()->
				GetCurrentPosition() );
		}

  } // end AfterEachIterationBase


  /**
	 * ********************* ConfigureImageSampler ************************
	 */

	template <class TElastix>
    void
    MetricBase<TElastix>
    ::ConfigureImageSampler( void )
  {
    /** Cast this to MetricWithSamplingType. */
    MetricWithSamplingType * thisAsMetricWithSampler
      = dynamic_cast< MetricWithSamplingType * >( this );

    if ( thisAsMetricWithSampler )
    {
      if ( thisAsMetricWithSampler->GetUseImageSampler() )
      {
        /** Typedefs of all available image samplers.
        * ImageFullSamplerType and ImageSamplerBaseType are already declared in the header. */
        typedef ImageRandomSampler< FixedImageType >              ImageRandomSamplerType;
        typedef ImageRandomSamplerSparseMask< FixedImageType >    ImageRandomSamplerSparseMaskType;
        typedef ImageRandomCoordinateSampler< FixedImageType >    ImageRandomCoordinateSamplerType;
        typedef ImageGridSampler< FixedImageType >                ImageGridSamplerType;

        /** Create an imageSampler of ImageSamplerBaseType. */
        typename ImageSamplerBaseType::Pointer imageSampler = 0;

        /** Get the desired sampler type from the parameter file. */
        unsigned int level =
			  ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
        std::string imageSamplerType = "Random";
        
        this->m_Configuration->ReadParameter( imageSamplerType, "ImageSampler",
          this->GetComponentLabel(), level, 0 );

        /** Get and set NumberOfSpatialSamples. This doesn't make sense for the ImageFullSampler. */
        unsigned long numberOfSpatialSamples = 5000;
        this->GetConfiguration()->ReadParameter( numberOfSpatialSamples, "NumberOfSpatialSamples",
            this->GetComponentLabel(), level, 0 );

        /** Set the imageSampler to the correct one. */
        if ( imageSamplerType == "Random" )
        {
          typename ImageRandomSamplerType::Pointer randomSampler
            = ImageRandomSamplerType::New();
          randomSampler->SetNumberOfSamples( numberOfSpatialSamples );
          imageSampler = randomSampler;
        }
        else if ( imageSamplerType == "RandomSparseMask" )
        {
          typename ImageRandomSamplerSparseMaskType::Pointer randomSamplerSparseMask
            = ImageRandomSamplerSparseMaskType::New();
          randomSamplerSparseMask->SetNumberOfSamples( numberOfSpatialSamples );
          imageSampler = randomSamplerSparseMask;
        }
        else if ( imageSamplerType == "Full" )
        {
          typename ImageFullSamplerType::Pointer fullSampler
            = ImageFullSamplerType::New();
          imageSampler = fullSampler;
        }
        else if ( imageSamplerType == "RandomCoordinate" )
        {
          typename ImageRandomCoordinateSamplerType::Pointer randomCoordinateSampler
            = ImageRandomCoordinateSamplerType::New();
          randomCoordinateSampler->SetNumberOfSamples( numberOfSpatialSamples );
          typedef typename ImageRandomCoordinateSamplerType::DefaultInterpolatorType
            FixedImageInterpolatorType;
          typename FixedImageInterpolatorType::Pointer fixedImageInterpolator =
            FixedImageInterpolatorType::New();
          /** Set the SplineOrder, default value = 3. */
	        unsigned int splineOrder = 3;
		      this->GetConfiguration()->ReadParameter( splineOrder,
            "FixedImageBSplineInterpolationOrder", this->GetComponentLabel(), level, 0 );
          fixedImageInterpolator->SetSplineOrder( splineOrder );
    		  randomCoordinateSampler->SetInterpolator(fixedImageInterpolator);
          imageSampler = randomCoordinateSampler;
        }
        else if ( imageSamplerType == "Grid" )
        {
          /** Create the gridSampler and the gridspacing. */
          typedef typename ImageGridSamplerType::SampleGridSpacingType        GridSpacingType;
          typedef typename ImageGridSamplerType::SampleGridSpacingValueType   SampleGridSpacingValueType;

          typename ImageGridSamplerType::Pointer gridSampler
            = ImageGridSamplerType::New();
          GridSpacingType gridspacing;

          /** Read the desired grid spacing of the samples. */
          unsigned int spacing_dim;
          for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
          {
            spacing_dim = 2;
            this->GetConfiguration()->ReadParameter( spacing_dim, "SampleGridSpacing", 
              this->GetComponentLabel(), level * FixedImageDimension + dim, -1 );
            gridspacing[ dim ] = static_cast<SampleGridSpacingValueType>( spacing_dim );
          }
          gridSampler->SetSampleGridSpacing( gridspacing );
          imageSampler = gridSampler;
        }
        else
        {
          itkExceptionMacro( << "ERROR: There exists no ImageSampler \"" << imageSamplerType << "\"." );
        }
     		
        /** Set the image sampler in the metric. */
        thisAsMetricWithSampler->SetImageSampler( imageSampler );

        /** Check if NewSamplesEveryIteration is possible with the selected ImageSampler. */
        std::string newSamples = "false";
        this->m_Configuration->ReadParameter( newSamples, "NewSamplesEveryIteration",
          this->GetComponentLabel(), level, 0, true );

        if ( newSamples == "true" )
        {
          bool ret = thisAsMetricWithSampler->GetImageSampler()->SelectNewSamplesOnUpdate();
          if ( !ret )
          {
            xl::xout["warning"]  << "WARNING: You want to select new samples every iteration, \
              but the selected ImageSampler is not suited for that." << std::endl;
          }
        }

      } // end if GetUseImageSampler

    } // end if

  } // end ConfigureImageSampler


	/**
	 * ********************* SelectNewSamples ************************
	 */

	template <class TElastix>
    void MetricBase<TElastix>::SelectNewSamples(void)
	{
    /** Cast this to MetricWithSamplingType. */
    MetricWithSamplingType * thisAsMetricWithSampler
      = dynamic_cast< MetricWithSamplingType * >( this );

    bool useSampler = false;

    if ( thisAsMetricWithSampler )
    {
      if ( thisAsMetricWithSampler->GetUseImageSampler() )
      {
        thisAsMetricWithSampler->GetImageSampler()->SelectNewSamplesOnUpdate();
        useSampler = true;
      }
    }
    if ( !useSampler )
    {
      /**
      * Force the metric to base its computation on a new subset of image samples.
      * Not every metric may have implemented this, so invoke an exception if this
      * method is called, without being overrided by a subclass.
      */
      xl::xout["error"]  << "ERROR: The SelectNewSamples function should be overridden or just not used." << std::endl;
      itkExceptionMacro( << "ERROR: The SelectNewSamples method is not implemented in your metric." );
    }

	} // end SelectNewSamples

  
  /**
	 * ********************* GetExactValue ************************
	 */

	template <class TElastix>
    typename MetricBase<TElastix>::MeasureType
    MetricBase<TElastix>::GetExactValue( const ParametersType& parameters )
  {
    /** Cast this to MetricWithSamplingType. */
    MetricWithSamplingType * thisAsMetricWithSampler
      = dynamic_cast< MetricWithSamplingType * >( this );

    /** Useless implementation if no image sampler is used; we may as
     * well throw an error, but the ShowExactMetricValue is not really
     * essential for good registration... */
    if ( thisAsMetricWithSampler == 0 )
    {
      return itk::NumericTraits<MeasureType>::Zero;
    }
    if ( thisAsMetricWithSampler->GetUseImageSampler() == false )
    {
      return itk::NumericTraits<MeasureType>::Zero;
    }

    /** Get the current image sampler */
    typename ImageSamplerBaseType::Pointer currentSampler = 
      thisAsMetricWithSampler->GetImageSampler();
    if ( currentSampler.IsNull() )
    {
      /** Again just a dummy implementation */
      return itk::NumericTraits<MeasureType>::Zero;
    }
    
    /** Try to cast the current Sampler to a FullSampler */
    ImageFullSamplerType * testPointer = 
      dynamic_cast<ImageFullSamplerType *>( currentSampler.GetPointer() );
    if ( testPointer != 0 )
    {
      /** GetValue gives us the exact value! */
      return this->GetAsITKBaseType()->GetValue(parameters);
    }
    
    /** We have to provide the metric a full sampler, calls its GetValue
     * and set back its original sampler */
    typedef typename ITKBaseType::FixedImageMaskType FixedImageMaskType;
    if ( this->m_ExactMetricSampler.IsNull() )
    {
      this->m_ExactMetricSampler = ImageFullSamplerType::New();
    }
    this->m_ExactMetricSampler->SetInput( currentSampler->GetInput() );
    this->m_ExactMetricSampler->SetMask( 
      const_cast< FixedImageMaskType * >( 
      dynamic_cast< const FixedImageMaskType * >(
      this->GetAsITKBaseType()->GetFixedImageMask() ) ) );
    this->m_ExactMetricSampler->SetInputImageRegion(
      currentSampler->GetInputImageRegion() );
    thisAsMetricWithSampler->SetImageSampler( this->m_ExactMetricSampler );
    
    /** Compute the metric value on the full images */
    MeasureType exactValue = 
      this->GetAsITKBaseType()->GetValue(parameters);
    
    /** reset the original sampler */
    thisAsMetricWithSampler->SetImageSampler( currentSampler );

    return exactValue;
        
  } // end GetExactValue


} // end namespace elastix


#endif // end #ifndef __elxMetricBase_hxx

