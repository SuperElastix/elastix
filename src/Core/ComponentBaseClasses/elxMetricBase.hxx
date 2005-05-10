#ifndef __elxMetricBase_hxx
#define __elxMetricBase_hxx

#include "elxMetricBase.h"

/** Mask support. */
#include "itkBinaryBallStructuringElement.h"
#include "itkGrayscaleErodeImageFilter.h"


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
			this->m_FixedMaskImageReader			= FixedMaskImageReaderType::New();
			this->m_FixedMaskAsImage					= FixedMaskImageType::New();
			this->m_FixedMaskAsSpatialObject	= FixedImageMaskSpatialObjectType::New();
			this->m_FixedMaskImageReader->SetFileName( fixedMaskFileName.c_str() );

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
			this->m_MovingMaskImageReader			= MovingMaskImageReaderType::New();
			this->m_MovingMaskAsImage					= MovingMaskImageType::New();
			this->m_MovingMaskAsSpatialObject	= MovingImageMaskSpatialObjectType::New();
			this->m_MovingMaskImageReader->SetFileName( movingMaskFileName.c_str() );

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

	} // end BeforeEachResolutionBase


	/**
	 * ************************* UpdateMasks ************************
	 */

	template <class TElastix>
  void MetricBase<TElastix>::UpdateMasks( unsigned int level )
	{
		/** Some typedef's. */
		typedef BinaryBallStructuringElement<
			MaskFilePixelType,
			MovingImageDimension >							StructuringElementType;
		typedef typename StructuringElementType::RadiusType		RadiusType;
		typedef GrayscaleErodeImageFilter<
			MovingMaskImageType,
			MovingMaskImageType,
			StructuringElementType >					ErodeFilterType;

		/** Erode and set the fixed mask if necessary. ****************************
		 **************************************************************************
		 */
		if ( this->GetAsITKBaseType()->GetFixedImageMask() )
		{
			/**
			 *  If more resolution levels are used, the image is subsampled. Before
			 *  subsampling the image is smoothed with a Gaussian filter, with variance
			 *  (schedule/2)^2. The 'schedule' depends on the resolution level.
			 *  The lowest resolution level has a schedule of 2^(nr_of_levels-1).
			 *  The 'radius' of the convolution filter is roughly twice the standard deviation.
			 *	 Thus, the parts in the edge with size 'radius' are influenced by the background.
			 */

			/** Create erosion-filters. */
			typename ErodeFilterType::Pointer erosionF[ FixedImageDimension ];
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				erosionF[ i ] = ErodeFilterType::New();
			}
      
			/** Declare radius-array and structuring element. */
			RadiusType								radiusarrayF;
			StructuringElementType		S_ballF;

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

			/** Convert image to spatial object and put it in the metric. */
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
			typename ErodeFilterType::Pointer erosionM[ MovingImageDimension ];
			for ( unsigned int i = 0; i < MovingImageDimension; i++ )
			{
				erosionM[ i ] = ErodeFilterType::New();
			}
      
			/** Declare radius-array and structuring element. */
			RadiusType								radiusarrayM;
			StructuringElementType		S_ballM;

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

			/** Convert image to spatial object and put it in the metric. */
			this->m_MovingMaskAsSpatialObject->SetImage( m_MovingMaskAsImage );
			this->GetAsITKBaseType()->SetMovingImageMask( m_MovingMaskAsSpatialObject );

		} // end if moving mask present

	} // end UpdateMasks


	/**
	 * ********************* SelectNewSamples ************************
	 */

	template <class TElastix>
  void MetricBase<TElastix>::SelectNewSamples(void)
	{
		/**
		 * Force the metric to base its computation on a new subset of image samples.
		 * Not every metric may have implemented this, so invoke an exception if this
		 * method is called, without being overrided by a subclass.
		 */

		xl::xout["error"] << "ERROR: The SelectNewSamples function should be overridden or just not used." << std::endl;
		itkExceptionMacro(<< "ERROR: The SelectNewSamples method is not implemented in your metric.");

	} // end SelectNewSamples

} // end namespace elastix


#endif // end #ifndef __elxMetricBase_hxx

