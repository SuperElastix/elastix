#ifndef __elxMutualInformationHistogramMetric_HXX__
#define __elxMutualInformationHistogramMetric_HXX__

#include "elxMutualInformationHistogramMetric.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MutualInformationHistogramMetric<TElastix>
		::MutualInformationHistogramMetric()
	{
		/** Initialize.*/
		this->m_FixedMaskImageReader = 0;
		this->m_MovingMaskImageReader = 0;

	} // end Constructor


	/**
	 * ************************ BeforeAll ***************************
	 */
	
	template <class TElastix>
		int MutualInformationHistogramMetric<TElastix>
		::BeforeAll(void)
	{
		/** Declare the return value and initialize it.*/
		int returndummy = 0;

		/** Check Command line options and print them to the logfile.*/
		elxout << "Command line options:" << std::endl;
		std::string check = "";

		/** Check for appearance of "-fMask".*/
		check = this->m_Configuration->GetCommandLineArgument( "-fMask" );
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
		return returndummy;

	} // end BeforeAll


	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MutualInformationHistogramMetric<TElastix>::
		Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of MutualInformationHistogramMetric metric took: "
			<< static_cast<long>(timer->GetElapsedClockSec() *1000) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void MutualInformationHistogramMetric<TElastix>::
		BeforeRegistration(void)
	{		
		/** Read masks if necessary.*/
		
		/** Read fixed mask.*/
		std::string fixedMaskFileName = this->m_Configuration->
			GetCommandLineArgument( "-fMask" );
		if ( !( fixedMaskFileName.empty() ) )
		{
			this->m_FixedMaskImageReader	= FixedMaskImageReaderType::New();
			this->m_FixedMaskImageReader->SetFileName( fixedMaskFileName.c_str() );

			/** Do the reading. */
			try
			{
				this->m_FixedMaskImageReader->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MutualInformationHistogramMetric - BeforeRegistration()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading fixed mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
			/** Set the fixedmask. */
			this->SetFixedMask( this->m_FixedMaskImageReader->GetOutput() );

		} // end if ( fixed mask present )
		
		/** Read moving mask. */
		std::string movingMaskFileName = this->m_Configuration->
			GetCommandLineArgument( "-mMask" );
		if ( !( movingMaskFileName.empty() ) )
		{
			this->m_MovingMaskImageReader	= MovingMaskImageReaderType::New();			
			this->m_MovingMaskImageReader->SetFileName( movingMaskFileName.c_str() );

			/** Do the reading. */
			try
			{
				this->m_MovingMaskImageReader->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MutualInformationHistogramMetric - BeforeRegistration()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading moving mask.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
			/** Set the movingmask. */
			this->SetMovingMask( this->m_MovingMaskImageReader->GetOutput() );

		} // end if ( moving mask present )
		
		/** \todo Select another kernel function than the Gaussian. */

	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MutualInformationHistogramMetric<TElastix>::
		BeforeEachResolution(void)
	{
		/** \todo adapt SecondOrderRegularisationMetric.
		 * Set alpha, which balances the similarity and deformation energy
		 * E_total = (1-alpha)*E_sim + alpha*E_def.
		 * metric->SetAlpha( config.GetAlpha(level) );
		 */

		/** Get the current resolution level.*/
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
		
		unsigned int nrOfParameters = this->m_Elastix->GetElxTransformBase()
			->GetAsITKBaseType()->GetNumberOfParameters();
		ScalesType derivativeStepLengthScales( nrOfParameters );
		derivativeStepLengthScales.Fill( 1.0 );

		/** Read the parameters from the ParameterFile.*
		this->m_Configuration->ReadParameter( histogramSize, "HistogramSize", 0 );
		this->m_Configuration->ReadParameter( paddingValue, "PaddingValue", 0 );
		this->m_Configuration->ReadParameter( derivativeStepLength, "DerivativeStepLength", 0 );
		this->m_Configuration->ReadParameter( derivativeStepLengthScales, "DerivativeStepLengthScales", 0 );
		this->m_Configuration->ReadParameter( upperBoundIncreaseFactor, "UpperBoundIncreaseFactor", 0 );
		this->m_Configuration->ReadParameter( usePaddingValue, "UsePaddingValue", 0 );
		*/
		/** Set them.*/
		//this->SetHistogramSize( ?? );
		//this->SetPaddingValue( ?? );
		//this->SetDerivativeStepLength( ?? );
		this->SetDerivativeStepLengthScales( derivativeStepLengthScales );
		//this->SetUpperBoundIncreaseFactor( ?? );
		//this->SetUsePaddingValue( ?? );
		
		/** Erode and Set masks if necessary.*/

		/** Read the number of resolutions from the ParameterFile.*/
		unsigned int numberOfResolutions = 3;
		this->m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
		
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
			this->SetFixedMask( (this->m_FixedMaskImageReader->GetOutput())->Erode( radius ) );

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
			this->SetMovingMask( (this->m_MovingMaskImageReader->GetOutput())->Erode( radius ) );

		} // end if movingmask present
		
	} // end BeforeEachResolution
	
	
} // end namespace elastix


#endif // end #ifndef __elxMutualInformationHistogramMetric_HXX__

