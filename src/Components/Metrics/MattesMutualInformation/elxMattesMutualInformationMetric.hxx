#ifndef __elxMattesMutualInformationMetric_HXX__
#define __elxMattesMutualInformationMetric_HXX__

#include "elxMattesMutualInformationMetric.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MattesMutualInformationMetric<TElastix>
		::MattesMutualInformationMetric()
	{
		/** Initialize.*/
		m_FixedMaskImageReader = 0;
		m_MovingMaskImageReader = 0;

		m_NewSamplesEveryIteration = false;
		m_ShowExactMetricValue = false;

	} // end Constructor


	/**
	 * ************************ BeforeAll ***************************
	 */
	
	template <class TElastix>
		int MattesMutualInformationMetric<TElastix>
		::BeforeAll(void)
	{
		/** Declare the return value and initialize it.*/
		int returndummy = 0;

		/** Check Command line options and print them to the logfile.*/
		elxout << "Command line options:" << std::endl;
		std::string check = "";

		/** Check for appearance of "-fMask".*/
		check = m_Configuration->GetCommandLineArgument( "-fMask" );
		if ( check == "" )
		{
			elxout << "-fMask\t\tunspecified, so no fixed mask used" << std::endl;
		}
		else
		{
			elxout << "-fMask\t\t" << check << std::endl;
		}

		/** Check for appearance of "-mMask".*/
		check = "";
		check = m_Configuration->GetCommandLineArgument( "-mMask" );
		if ( check == "" )
		{
			elxout << "-mMask\t\tunspecified, so no moving mask used" << std::endl;
		}
		else
		{
			elxout << "-mMask\t\t" << check << std::endl;
		}

		/** Return a value.*/
		return returndummy;

	} // end BeforeAll


	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of MattesMutualInformation metric took: "
			<< static_cast<long>(timer->GetElapsedClockSec() *1000) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::BeforeRegistration(void)
	{		
		/** Read masks if necessary.*/
		
		/** Read fixed mask.*/
		std::string fixedMaskFileName = m_Configuration->
			GetCommandLineArgument( "-fMask" );
		if ( !( fixedMaskFileName.empty() ) )
		{
			m_FixedMaskImageReader	= FixedMaskImageReaderType::New();
			m_FixedMaskImageReader->SetFileName( fixedMaskFileName.c_str() );

			/** Do the reading. */
			try
			{
				m_FixedMaskImageReader->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MattesMutualInformationMetric - BeforeRegistration()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading fixed mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
			/** Set the fixedmask.*/
			this->SetFixedMask( m_FixedMaskImageReader->GetOutput() );

		} // end if ( fixed mask present )
		
		/** Read moving mask. */
		std::string movingMaskFileName = m_Configuration->
			GetCommandLineArgument( "-mMask" );
		if ( !( movingMaskFileName.empty() ) )
		{
			m_MovingMaskImageReader	= MovingMaskImageReaderType::New();
			m_MovingMaskImageReader->SetFileName( movingMaskFileName.c_str() );

			/** Do the reading. */
			try
			{
				m_MovingMaskImageReader->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MattesMutualInformationMetric - BeforeRegistration()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading moving mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
			/** Set the movingmask. */
			this->SetMovingMask( m_MovingMaskImageReader->GetOutput() );

		} // end if ( moving mask present )
	
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** \todo Adapt SecondOrderRegularisationMetric.
		 * Set alpha, which balances the similarity and deformation energy
		 * E_total = (1-alpha)*E_sim + alpha*E_def.
		 * 	metric->SetAlpha( config.GetAlpha(level) );
		 */

		/** Get the current resolution level.*/
		unsigned int level = 
			( m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
		
		/** Set the number of histogram bins and spatial samples.*/				
		unsigned int numberOfHistogramBins = 32;
		unsigned int numberOfSpatialSamples = 10000;
		/** \todo guess the default numberOfSpatialSamples from the 
		 * imagesize, the numberOfParameters, and the number of bins...
		 */
		
		/** Read the parameters from the ParameterFile.*/
		m_Configuration->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", level );
		m_Configuration->ReadParameter( numberOfSpatialSamples, "NumberOfSpatialSamples", level );
		
		/** Set them.*/
		this->SetNumberOfHistogramBins( numberOfHistogramBins );
		this->SetNumberOfSpatialSamples( numberOfSpatialSamples );
		
		/** Erode and Set masks if necessary.*/

		/** Read the number of resolutions from the ParameterFile.*/
		unsigned int numberOfResolutions = 3;
		m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
		
		/** Erode and Set the fixed Mask if necessary */
		if ( this->GetFixedMask() )
		{
			/**
			*  If more resolution levels are used, the image is subsampled. Before
			*  subsampling the image is smoothed with a Gaussian filter, with variance
			*  (schedule/2)^2. The 'schedule' depends on the resolution level.
			*  The lowest resolution level has a schedule of 2^(nr_of_levels-1).
			*  The 'radius' of the convolution filter is roughly twice the standard deviation.
			*	 Thus, the parts in the edge with size 'radius' are influenced by the background.
			*/
		
			/** Define the radius.*/
			unsigned long radius = static_cast<unsigned long>(
				ceil( pow( 2.0, static_cast<int>(	numberOfResolutions - level - 1 ) ) ) + 1 );

			/** Erode the mask.*/
			this->SetFixedMask( (m_FixedMaskImageReader->GetOutput())->Erode( radius ) );

		} // end if fixedmask present

		/** Erode and Set the moving Mask if necessary */
		if ( this->GetMovingMask() )
		{
			/**
			*	Same story as before. Now the size the of the eroding element is doubled.
			* This is because the gradient of the moving image is used for calculating
			* the derivative of the metric. 
			*/
			
			/** Define the radius.*/
			unsigned long radius = static_cast<unsigned long>(
				ceil( pow( 2.0, static_cast<int>(	numberOfResolutions - level ) ) ) + 1 );

			/** Erode the mask.*/
			this->SetMovingMask( (m_MovingMaskImageReader->GetOutput())->Erode( radius ) );

		} // end if movingmask present


		/** Check if the exact metric value, computed on all pixels, should be shown, 
		 * and whether the exact metric derivative should be used
		 */

		/** Remove the ExactMetric-column, if it already existed. */
		xl::xout["iteration"].RemoveTargetCell("ExactMetric");

		bool useExactDerivativeBool = false;
		std::string useExactDerivative = "false";
		this->GetConfiguration()->
			ReadParameter(useExactDerivative, "UseExactMetricDerivative", level);
		if (useExactDerivative == "true")
		{
			useExactDerivativeBool = true;
		}
		else
		{
			useExactDerivativeBool = false;
		}
		this->SetUseExactDerivative(useExactDerivativeBool);

		if (!useExactDerivativeBool)
		{
			/** Show the exact metric VALUE anyway? */
			std::string showExactMetricValue = "false";
			this->GetConfiguration()->
				ReadParameter(showExactMetricValue, "ShowExactMetricValue", level);
			if (showExactMetricValue == "true")
			{
				m_ShowExactMetricValue = true;
				xl::xout["iteration"].AddTargetCell("ExactMetric");
				xl::xout["iteration"]["ExactMetric"] << std::showpoint << std::fixed;
			}
			else
			{
				m_ShowExactMetricValue = false;
			}

			/** Check if after every iteration a new sample set should be created */
			std::string newSamplesEveryIteration = "false";
			this->GetConfiguration()->
				ReadParameter(newSamplesEveryIteration, "NewSamplesEveryIteration", level);
			if (newSamplesEveryIteration == "true")
			{	
				m_NewSamplesEveryIteration = true;
			}
			else
			{
				m_NewSamplesEveryIteration = false;
			}
			
			/** Check if the "smart sample strategy" should be used */
			std::string smartSampleSelect = "false";
			this->GetConfiguration()->
				ReadParameter(smartSampleSelect,"SmartSampleSelect", level);
			if (smartSampleSelect == "true")
			{
				this->SetSmartSampleSelect(true);
			}
			else
			{
				this->SetSmartSampleSelect(false);
			}

		}
		else	
		{
			/** The exact metric value is shown anyway */
			m_ShowExactMetricValue = false;

			/** And new samples every iteration is not appropriate either */
			m_NewSamplesEveryIteration = false;
		}
		
	} // end BeforeEachResolution
	


	/**
	 * ***************AfterEachIteration ****************************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::AfterEachIteration(void)
	{		
		/** Show the mutual information computed on all voxels,
		 * if the user wanted it */
		if (m_ShowExactMetricValue)
		{
			xl::xout["iteration"]["ExactMetric"] << this->GetExactValue(
				this->GetElastix()->
				GetElxOptimizerBase()->GetAsITKBaseType()->
				GetCurrentPosition() );
		}

		/** Create a new sample set if the user wanted it  */
		if (m_NewSamplesEveryIteration)
		{
			this->SampleFixedImageDomain();
		}
	}

	
} // end namespace elastix


#endif // end #ifndef __elxMattesMutualInformationMetric_HXX__

