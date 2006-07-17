#ifndef __elxMetricBase_hxx
#define __elxMetricBase_hxx

#include "elxMetricBase.h"

/** Mask support. */
#include "itkBinaryBallStructuringElement.h"
#include "itkGrayscaleErodeImageFilter.h"

#include "itkImageSamplerBase.h"
#include "itkImageRandomSampler.h"
#include "itkImageFullSampler.h"
//#include "itkImageRandomCoordinateSampler.h"
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
		this->m_FixedMaskImageReader = 0;
		this->m_MovingMaskImageReader = 0;
		this->m_FixedMaskAsImage = 0;
		this->m_MovingMaskAsImage = 0;
		this->m_FixedMaskAsSpatialObject = 0;
		this->m_MovingMaskAsSpatialObject = 0;

	} // end Constructor

		
	/**
	 * ************************ BeforeAllBase ***************************
	 */
	
	template <class TElastix>
		int MetricBase<TElastix>
		::BeforeAllBase(void)
	{
		/** Check Command line options and print them to the logfile. */
		elxout << "Command line options from MetricBase:" << std::endl;
		std::string check = "";

		/** Check for appearance of "-fMask". */
		check = this->m_Configuration->GetCommandLineArgument( "-fMask" );
		if ( check == "" )
		{
			elxout << "-fMask\t\tunspecified, so no fixed mask used" << std::endl;
		}
		else
		{
			elxout << "-fMask\t\t" << check << std::endl;
		}

		/** Check for appearance of "-mMask". */
		check = "";
		check = this->m_Configuration->GetCommandLineArgument( "-mMask" );
		if ( check == "" )
		{
			elxout << "-mMask\t\tunspecified, so no moving mask used" << std::endl;
		}
		else
		{
			elxout << "-mMask\t\t" << check << std::endl;
		}

		/** Return a value.*/
		return 0;

	} // end BeforeAllBase


	/**
	 * ******************* BeforeRegistrationBase ********************
	 */

	template <class TElastix>
		void MetricBase<TElastix>::
		BeforeRegistrationBase(void)
	{
		/** Read masks if necessary and start a timer. */
		TimerPointer timer = TimerType::New();
		timer->StartTimer();

		/** Read fixed mask.*/
		std::string fixedMaskFileName = this->m_Configuration->
			GetCommandLineArgument( "-fMask" );
		if ( !( fixedMaskFileName.empty() ) )
		{
			/** Create reader for fixed mask. */
			this->m_FixedMaskImageReader			= FixedMaskImageReaderType::New();
			this->m_FixedMaskImageReader->SetFileName( fixedMaskFileName.c_str() );
			/** Create spatial object for fixed mask. */
			this->m_FixedMaskAsSpatialObject	= FixedImageMaskSpatialObjectType::New();

			/** Do the reading. */
			try
			{
				this->m_FixedMaskImageReader->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MetricMetric - BeforeRegistrationBase()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading fixed mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

			/** Set the fixed mask. */
			this->m_FixedMaskAsImage = this->m_FixedMaskImageReader->GetOutput();
			this->m_FixedMaskAsSpatialObject->SetImage( this->m_FixedMaskAsImage );
			this->GetAsITKBaseType()->SetFixedImageMask( m_FixedMaskAsSpatialObject );

		} // end if ( fixed mask present )
		
		/** Read moving mask. */
		std::string movingMaskFileName = this->m_Configuration->
			GetCommandLineArgument( "-mMask" );
		if ( !( movingMaskFileName.empty() ) )
		{
			/** Create reader for moving mask. */
			this->m_MovingMaskImageReader			= MovingMaskImageReaderType::New();
			this->m_MovingMaskImageReader->SetFileName( movingMaskFileName.c_str() );
			/** Create spatial object for moving mask. */
			this->m_MovingMaskAsSpatialObject	= MovingImageMaskSpatialObjectType::New();
			
			/** Do the reading. */
			try
			{
				this->m_MovingMaskImageReader->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MetricBase - BeforeRegistrationBase()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading moving mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

			/** Set the moving mask. */
			this->m_MovingMaskAsImage = this->m_MovingMaskImageReader->GetOutput();
			this->m_MovingMaskAsSpatialObject->SetImage( this->m_MovingMaskAsImage );
			this->GetAsITKBaseType()->SetMovingImageMask( m_MovingMaskAsSpatialObject );

		} // end if ( moving mask present )

		/** If there are masks, print the elapsed time for reading them. */
		timer->StopTimer();
		if ( this->GetAsITKBaseType()->GetFixedImageMask() ||
			this->GetAsITKBaseType()->GetMovingImageMask() )
		{
			elxout << "Reading the masks took: "
				<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;
		}

	} // end BeforeRegistrationBase


	/**
	 * ******************* BeforeEachResolutionBase ******************
	 */

	template <class TElastix>
		void MetricBase<TElastix>
		::BeforeEachResolutionBase(void)
	{
		/** Create timer. */
		TimerPointer timer = TimerType::New();

		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

		/** Get from the parameterfile if erosions are wanted.
		 * NOTE: Erosion of the mask at lower resolutions, prevents
		 * the border / edge of the mask taken into account.
		 * This can be usefull for example for ultrasound images,
		 * where you don't want to take into account values outside
		 * the US-beam, but where you also don't want to match the
		 * edge / border of this beam.
		 * For example for MRI's of the head, the borders of the head
		 * may be wanted to match, and there erosion should be avoided.
		 */
		std::string erosionOrNot = "true";
		this->m_Configuration->ReadParameter( erosionOrNot, "ErodeMask", 0 );

		/** If there are any masks and if wanted, update them by erosion. */
		if ( this->GetAsITKBaseType()->GetFixedImageMask() ||
			this->GetAsITKBaseType()->GetMovingImageMask() )
		{
			if ( erosionOrNot == "true" )
			{
				/** Start timer. */
				timer->StartTimer();

				/** Erode masks. */
				this->UpdateMasks( level );

				/** Stop timer and print the elapsed time. */
				timer->StopTimer();
				elxout << "Eroding the masks took: "
					<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;
			}
		}

    /** Configure the image sampler. */
    this->ConfigureImageSampler();

	} // end BeforeEachResolutionBase


	/**
	 * ************************* UpdateMasks ************************
	 */

	template <class TElastix>
  void MetricBase<TElastix>::UpdateMasks( unsigned int level )
	{

		/**\todo: moet dit eigenlijk niet in de fixed/moving-pyramidbase ofzo? */

		/** Some typedef's. */
		typedef BinaryBallStructuringElement<
			MaskFilePixelType,
			FixedImageDimension >							StructuringElementTypeF;
		typedef typename StructuringElementTypeF::RadiusType		RadiusTypeF;
		typedef GrayscaleErodeImageFilter<
			FixedMaskImageType,
			FixedMaskImageType,
			StructuringElementTypeF >					ErodeFilterTypeF;
		typedef BinaryBallStructuringElement<
			MaskFilePixelType,
			MovingImageDimension >							StructuringElementTypeM;
		typedef typename StructuringElementTypeM::RadiusType		RadiusTypeM;
		typedef GrayscaleErodeImageFilter<
			MovingMaskImageType,
			MovingMaskImageType,
			StructuringElementTypeM >					ErodeFilterTypeM;

		/** Erode and set the fixed mask if necessary. ****************************
		 **************************************************************************
		 */
		if ( this->GetAsITKBaseType()->GetFixedImageMask() )
		{
			/**
			 *  If more resolution levels are used, the image is subsampled. Before
			 *  subsampling the image is smoothed with a Gaussian filter, with variance
			 *  (schedule/2)^2. The 'schedule' depends on the resolution level.
			 *  The 'radius' of the convolution filter is roughly twice the standard deviation.
			 *	Thus, the parts in the edge with size 'radius' are influenced by the background.
			 */

			/** Create erosion-filters. */
			typename ErodeFilterTypeF::Pointer erosionF[ FixedImageDimension ];
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				erosionF[ i ] = ErodeFilterTypeF::New();
			}
      
			/** Declare radius-array and structuring element. */
			RadiusTypeF								radiusarrayF;
			StructuringElementTypeF		S_ballF;
			
			/** Setup the erosion pipeline. */
			erosionF[ 0 ]->SetInput( this->m_FixedMaskImageReader->GetOutput() );
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				/** Create the radius array. */
				radiusarrayF.Fill( 0 );
				unsigned int schedule = this->GetElastix()->GetElxFixedImagePyramidBase()
					->GetAsITKBaseType()->GetSchedule()[ level ][ i ];
				unsigned long radius = static_cast<unsigned long>( schedule + 1 );
				radiusarrayF.SetElement( i, radius );

				/** Create the structuring element and set it into the erosion filter. */
				S_ballF.SetRadius( radiusarrayF );
				S_ballF.CreateStructuringElement();
				erosionF[ i ]->SetKernel( S_ballF );
								
				/** Connect the pipeline. */
				if ( i > 0 ) erosionF[ i ]->SetInput( erosionF[ i - 1 ]->GetOutput() );			
			}

			/** Set output of the erosion to m_FixedImageMaskAsImage. */
			this->m_FixedMaskAsImage = erosionF[ FixedImageDimension - 1 ]->GetOutput();
			this->m_FixedMaskAsImage->Modified();
			
			/** Do the erosion. */
			try
			{
				this->m_FixedMaskAsImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MetricBase - UpdateMasks()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError while eroding the fixed mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

			/** Convert image to spatial object and put it in the metric.
			 * NOTE: It is important to destruct the old SpatialObject
			 * and create a new one, because otherwise in SetImage()
			 * the spacing gets multiplied and multiplied, and so on,
			 * for each time this function is called (i.e. each resolution).
			 */
			this->m_FixedMaskAsSpatialObject = FixedImageMaskSpatialObjectType::New();
			this->m_FixedMaskAsSpatialObject->SetImage( m_FixedMaskAsImage );
			this->GetAsITKBaseType()->SetFixedImageMask( m_FixedMaskAsSpatialObject );

		} // end if fixed mask present

		/** Erode and set the moving mask if necessary. ***************************
		 **************************************************************************
		 */
		if ( this->GetAsITKBaseType()->GetMovingImageMask() )
		{
			/**
			 *	Same story as before. Now the size the of the eroding element is doubled.
			 * This is because the gradient of the moving image is used for calculating
			 * the derivative of the metric. 
			 */

			/** Create erosion-filters. */
			typename ErodeFilterTypeM::Pointer erosionM[ MovingImageDimension ];
			for ( unsigned int i = 0; i < MovingImageDimension; i++ )
			{
				erosionM[ i ] = ErodeFilterTypeM::New();
			}
      
			/** Declare radius-array and structuring element. */
			RadiusTypeM								radiusarrayM;
			StructuringElementTypeM		S_ballM;
			
			/** Setup the erosion pipeline. */
			erosionM[ 0 ]->SetInput( this->m_MovingMaskImageReader->GetOutput() );
			for ( unsigned int i = 0; i < MovingImageDimension; i++ )
			{
				/** Create the radius array. */
				radiusarrayM.Fill( 0 );
				unsigned int schedule = this->GetElastix()->GetElxMovingImagePyramidBase()
					->GetAsITKBaseType()->GetSchedule()[ level ][ i ];
				unsigned long radius = static_cast<unsigned long>( 2 * schedule + 1 );
				radiusarrayM.SetElement( i, radius );

				/** Create the structuring element and set it into the erosion filter. */
				S_ballM.SetRadius( radiusarrayM );
				S_ballM.CreateStructuringElement();
				erosionM[ i ]->SetKernel( S_ballM );
								
				/** Connect the pipeline. */
				if ( i > 0 ) erosionM[ i ]->SetInput( erosionM[ i - 1 ]->GetOutput() );			
			}

			/** Set output of the erosion to m_MovingImageMaskAsImage. */
			this->m_MovingMaskAsImage = erosionM[ MovingImageDimension - 1 ]->GetOutput();
			this->m_MovingMaskAsImage->Modified();
			
			/** Do the erosion. */
			try
			{
				this->m_MovingMaskAsImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MetricBase - UpdateMasks()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError while eroding the moving mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

			/** Convert image to spatial object and put it in the metric.
			 * NOTE: It is important to destruct the old SpatialObject
			 * and create a new one, because otherwise in SetImage()
			 * the spacing gets multiplied and multiplied, and so on,
			 * for each time this function is called (i.e. each resolution).
			 */
			this->m_MovingMaskAsSpatialObject = MovingImageMaskSpatialObjectType::New();
			this->m_MovingMaskAsSpatialObject->SetImage( m_MovingMaskAsImage );
			this->GetAsITKBaseType()->SetMovingImageMask( m_MovingMaskAsSpatialObject );

		} // end if moving mask present

	} // end UpdateMasks


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
      /** Typedefs of all available image samplers. */
      typedef ImageSamplerBase< FixedImageType >      ImageSamplerBaseType;
      typedef ImageRandomSampler< FixedImageType >    ImageRandomSamplerType;
      typedef ImageFullSampler< FixedImageType >      ImageFullSamplerType;
      //typedef ImageRandomCoordinateSampler< FixedImageType >      ImageRandomCoordinateSamplerType;
      typedef ImageGridSampler< FixedImageType >      ImageGridSamplerType;

      /** Create an imageSampler of ImageSamplerBaseType. */
      typename ImageSamplerBaseType::Pointer imageSampler = 0;

      /** Get the desired sampler type from the parameter file. */
      unsigned int level =
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
      std::string imageSamplerType = "Random";
      this->m_Configuration->ReadParameter( imageSamplerType, "ImageSampler", 0 );
      this->m_Configuration->ReadParameter( imageSamplerType, "ImageSampler", level );

      /** Get and set NumberOfSpatialSamples. This doesn't make sense for the ImageFullSampler. */
      unsigned long numberOfSpatialSamples = 5000;
      this->GetConfiguration()->ReadParameter( numberOfSpatialSamples, "NumberOfSpatialSamples", 0 );
      this->GetConfiguration()->ReadParameter( numberOfSpatialSamples, "NumberOfSpatialSamples", level );

      /** Set the imageSampler to the correct one. */
      if ( imageSamplerType == "Random" )
      {
        typename ImageRandomSamplerType::Pointer randomSampler
          = ImageRandomSamplerType::New();
        randomSampler->SetNumberOfSamples( numberOfSpatialSamples );
        imageSampler = randomSampler;
      }
      else if ( imageSamplerType == "Full" )
      {
        typename ImageFullSamplerType::Pointer fullSampler
          = ImageFullSamplerType::New();
        imageSampler = fullSampler;
      }
      /*else if ( imageSamplerType == "RandomCoordinate" )
      {
        typename ImageRandomCoordinateSamplerType::Pointer randomcoordinateSampler
          = ImageRandomCoordinateSamplerType::New();
        randomcoordinateSampler->SetNumberOfSamples( numberOfSpatialSamples );
        imageSampler = randomcoordinateSampler;
      }*/
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
          this->GetConfiguration()->ReadParameter(
            spacing_dim, "SampleGridSpacing", level * FixedImageDimension + dim );
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
      this->m_Configuration->ReadParameter( newSamples, "NewSamplesEveryIteration", 0, true );
      this->m_Configuration->ReadParameter( newSamples, "NewSamplesEveryIteration", level, true );

      if ( newSamples == "true" )
      {
        bool ret = thisAsMetricWithSampler->GetImageSampler()->SelectNewSamplesOnUpdate();
        if ( !ret )
        {
          xl::xout["warning"]  << "WARNING: You want to select new samples every iteration, \
                                  but the selected ImageSampler is not suited for that." << std::endl;
        }
      }
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

    if ( thisAsMetricWithSampler )
    {
      thisAsMetricWithSampler->GetImageSampler()->SelectNewSamplesOnUpdate();
    }
    else
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


} // end namespace elastix


#endif // end #ifndef __elxMetricBase_hxx

