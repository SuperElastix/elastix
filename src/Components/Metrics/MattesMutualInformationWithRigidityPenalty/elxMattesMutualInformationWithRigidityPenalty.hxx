#ifndef __elxMattesMutualInformationWithRigidityPenalty_HXX__
#define __elxMattesMutualInformationWithRigidityPenalty_HXX__

#include "elxMattesMutualInformationWithRigidityPenalty.h"
#include "vnl/vnl_math.h"
#include <string>

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MattesMutualInformationWithRigidityPenalty<TElastix>
		::MattesMutualInformationWithRigidityPenalty()
	{
		this->m_FixedRigidityImageReader = 0;
		this->m_MovingRigidityImageReader = 0;

		/** Initialize m_RigidityPenaltyWeightVector to be 1.0 for each resolution. */
		this->m_RigidityPenaltyWeightVector.resize( 1, 1.0 );

		/** Initialize m_DilateRigidityImagesVector to be true for each resolution. */
		this->m_DilateRigidityImagesVector.resize( 1, true );

	} // end Constructor


	/**
	 * ********************** BeforeRegistration *********************
	 */
	
	template <class TElastix>
		void MattesMutualInformationWithRigidityPenalty<TElastix>
		::BeforeRegistration(void)
	{
		/** Get the number of resolution levels. */
		unsigned int numberOfResolutions = 3;
		this->GetConfiguration()->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );

		/** Get and set the RigidityPenaltyWeight. */
		this->GetConfiguration()->ReadParameter( this->m_RigidityPenaltyWeightVector[ 0 ], "RigidityPenaltyWeight", 0 );
		this->m_RigidityPenaltyWeightVector.resize( numberOfResolutions, this->m_RigidityPenaltyWeightVector[ 0 ] );
		for ( unsigned int i = 1; i < numberOfResolutions; i++ )
		{
      this->m_Configuration->ReadParameter( this->m_RigidityPenaltyWeightVector[ i ], "RigidityPenaltyWeight", i, true );
		}
		/** Set the RigidityPenaltyWeight in the superclass to the first resolution weight. */
    this->SetMetricWeight( this->m_RigidityPenaltyWeightVector[ 0 ], 1 );

		/** Get and set the linearityConditionWeight. */
		double linearityConditionWeight = 1.0;
		this->GetConfiguration()->ReadParameter( linearityConditionWeight, "LinearityConditionWeight", 0 );
		this->m_RigidityPenaltyTermMetric->SetLinearityConditionWeight( linearityConditionWeight );

		/** Get and set the orthonormalityConditionWeight. */
		double orthonormalityConditionWeight = 1.0;
		this->GetConfiguration()->ReadParameter( orthonormalityConditionWeight, "OrthonormalityConditionWeight", 0 );
		this->m_RigidityPenaltyTermMetric->SetOrthonormalityConditionWeight( orthonormalityConditionWeight );

		/** Get and set the propernessWeight. */
		double propernessConditionWeight = 1.0;
		this->GetConfiguration()->ReadParameter( propernessConditionWeight, "PropernessConditionWeight", 0 );
		this->m_RigidityPenaltyTermMetric->SetPropernessConditionWeight( propernessConditionWeight );

    /** Set the usage of the linearity condition part. */
    std::string useLinearityCondition = "true";
    bool useLinearityConditionBool = true;
    this->GetConfiguration()->ReadParameter( useLinearityCondition, "UseLinearityCondition", 0 );
    if ( useLinearityCondition == "false" ) useLinearityConditionBool = false;
    this->m_RigidityPenaltyTermMetric->SetUseLinearityCondition( useLinearityConditionBool );

    /** Set the usage of the orthonormality condition part. */
    std::string useOrthonormalityCondition = "true";
    bool useOrthonormalityConditionBool = true;
    this->GetConfiguration()->ReadParameter( useOrthonormalityCondition, "UseOrthonormalityCondition", 0 );
    if ( useOrthonormalityCondition == "false" ) useOrthonormalityConditionBool = false;
    this->m_RigidityPenaltyTermMetric->SetUseOrthonormalityCondition( useOrthonormalityConditionBool );

    /** Set the usage of the properness condition part. */
    std::string usePropernessCondition = "true";
    bool usePropernessConditionBool = true;
    this->GetConfiguration()->ReadParameter( usePropernessCondition, "UsePropernessCondition", 0 );
    if ( usePropernessCondition == "false" ) usePropernessConditionBool = false;
    this->m_RigidityPenaltyTermMetric->SetUsePropernessCondition( usePropernessConditionBool );

    /** Set the calculation of the linearity condition part. */
    std::string calculateLinearityCondition = "true";
    bool calculateLinearityConditionBool = true;
    this->GetConfiguration()->ReadParameter( calculateLinearityCondition, "CalculateLinearityCondition", 0 );
    if ( calculateLinearityCondition == "false" ) calculateLinearityConditionBool = false;
    this->m_RigidityPenaltyTermMetric->SetCalculateLinearityCondition( calculateLinearityConditionBool );

    /** Set the calculation of the orthonormality condition part. */
    std::string calculateOrthonormalityCondition = "true";
    bool calculateOrthonormalityConditionBool = true;
    this->GetConfiguration()->ReadParameter( calculateOrthonormalityCondition, "CalculateOrthonormalityCondition", 0 );
    if ( calculateOrthonormalityCondition == "false" ) calculateOrthonormalityConditionBool = false;
    this->m_RigidityPenaltyTermMetric->SetCalculateOrthonormalityCondition( calculateOrthonormalityConditionBool );

    /** Set the calculation of the properness condition part. */
    std::string calculatePropernessCondition = "true";
    bool calculatePropernessConditionBool = true;
    this->GetConfiguration()->ReadParameter( calculatePropernessCondition, "CalculatePropernessCondition", 0 );
    if ( calculatePropernessCondition == "false" ) calculatePropernessConditionBool = false;
    this->m_RigidityPenaltyTermMetric->SetCalculatePropernessCondition( calculatePropernessConditionBool );

		/** Get and set the m_DilateRigidityImagesVector. */
		std::string tmp = "true";
		this->GetConfiguration()->ReadParameter( tmp, "DilateRigidityImages", 0 );
		std::vector< std::string > dilateRigidityImagesVector( numberOfResolutions, tmp );
		if ( tmp == "false" ) this->m_DilateRigidityImagesVector[ 0 ] = false;
		this->m_DilateRigidityImagesVector.resize( numberOfResolutions );
		for ( unsigned int i = 1; i < numberOfResolutions; i++ )
		{
      this->m_Configuration->ReadParameter( dilateRigidityImagesVector[ i ], "DilateRigidityImages", i );
			if ( dilateRigidityImagesVector[ i ] == "true" ) this->m_DilateRigidityImagesVector[ i ] = true;
			else this->m_DilateRigidityImagesVector[ i ] = false;
		}
		/** Set the DilateRigidityImages in the superclass to the first resolution option. */
		this->SetDilateRigidityImages( this->m_DilateRigidityImagesVector[ 0 ] );

		/** Get and set the dilationRadiusMultiplier. */
		double dilationRadiusMultiplier = 1.0;
		this->GetConfiguration()->ReadParameter( dilationRadiusMultiplier, "DilationRadiusMultiplier", 0 );
		this->SetDilationRadiusMultiplier( dilationRadiusMultiplier );

		/** Get and set the output directory name. *
		std::string outdir = this->GetConfiguration()->GetCommandLineArgument( "-out" );
		this->SetOutputDirectoryName( outdir.c_str() );

		/** Get and set the useFixedRigidityImage and read the FixedRigidityImage if wanted. */
		std::string useFixedRigidityImage = "true";
		this->GetConfiguration()->ReadParameter( useFixedRigidityImage, "UseFixedRigidityImage", 0 );
		if ( useFixedRigidityImage == "true" )
		{
			/** Use the FixedRigidityImage. */
			this->SetUseFixedRigidityImage( true );

			/** Read the fixed rigidity image and set it in the right class. */
			std::string fixedRigidityImageName = "";
			this->GetConfiguration()->ReadParameter( fixedRigidityImageName, "FixedRigidityImageName", 0 );

			/** Check if a name is given. */
			if ( fixedRigidityImageName == "" )
			{
				/** Create and throw an exception. */
				itkExceptionMacro( << "ERROR: No fixed rigidity image filename specified." );
			}
			else
			{
				/** Create the reader and set the filename. */
				this->m_FixedRigidityImageReader = RigidityImageReaderType::New();
				this->m_FixedRigidityImageReader->SetFileName( fixedRigidityImageName.c_str() );

				/** Do the reading. */
				try
				{
					this->m_FixedRigidityImageReader->Update();
				}
				catch( ExceptionObject & excp )
				{
					/** Add information to the exception. */
					excp.SetLocation( "MattesMutualInformationWithRigidityPenalty - BeforeEachResolution()" );
					std::string err_str = excp.GetDescription();
					err_str += "\nError occured while reading the FixedRigidityImage.\n";
					excp.SetDescription( err_str );
					/** Pass the exception to an higher level. */
					throw excp;
				}

				/** Set the fixed rigidity image into the superclass. */
				this->SetFixedRigidityImage( this->m_FixedRigidityImageReader->GetOutput() );
        
			} // end if filename
		}
		else
		{
			this->SetUseFixedRigidityImage( false );
		} // end if use fixedRigidityImage

		/** Get and set the useMovingRigidityImage and read the movingRigidityImage if wanted. */
		std::string useMovingRigidityImage = "true";
		this->GetConfiguration()->ReadParameter( useMovingRigidityImage, "UseMovingRigidityImage", 0 );
		if ( useMovingRigidityImage == "true" )
		{
			/** Use the movingRigidityImage. */
			this->SetUseMovingRigidityImage( true );
			
			/** Read the moving rigidity image and set it in the right class. */
			std::string movingRigidityImageName = "";
			this->GetConfiguration()->ReadParameter( movingRigidityImageName, "MovingRigidityImageName", 0 );
      
			/** Check if a name is given. */
			if ( movingRigidityImageName == "" )
			{
				/** Create and throw an exception. */
				itkExceptionMacro( << "ERROR: No moving rigidity image filename specified." );
			}
			else
			{
				/** Create the reader and set the filename. */
				this->m_MovingRigidityImageReader = RigidityImageReaderType::New();
				this->m_MovingRigidityImageReader->SetFileName( movingRigidityImageName.c_str() );
        
				/** Do the reading. */
				try
				{
					this->m_MovingRigidityImageReader->Update();
				}
				catch( ExceptionObject & excp )
				{
					/** Add information to the exception. */
					excp.SetLocation( "MattesMutualInformationWithRigidityPenalty - BeforeEachResolution()" );
					std::string err_str = excp.GetDescription();
					err_str += "\nError occured while reading the MovingRigidityImage.\n";
					excp.SetDescription( err_str );
					/** Pass the exception to an higher level. */
					throw excp;
				}

				/** Set the moving rigidity image into the superclass. */
				this->SetMovingRigidityImage( this->m_MovingRigidityImageReader->GetOutput() );

			} // end if filename
		}
		else
		{
			this->SetUseMovingRigidityImage( false );
		} // end if use movingRigidityImage

		/** Important check: at least one rigidity image must be given. */
		if ( useFixedRigidityImage == "false" && useMovingRigidityImage == "false" )
		{
      xl::xout["warning"] << "WARNING: UseFixedRigidityImage and UseMovingRigidityImage are both true.\n";
      xl::xout["warning"] << "         The rigidity penalty term is evaluated on entire input transform domain." << std::endl;
		}

		/** Add target cells to xout["iteration"]. */
    xout["iteration"].AddTargetCell("5:Metric-MI");
    xout["iteration"].AddTargetCell("6:Metric-RP");
    xout["iteration"].AddTargetCell("7:Metric-LC");
    xout["iteration"].AddTargetCell("8:Metric-OC");
    xout["iteration"].AddTargetCell("9:Metric-PC");

		/** Format the metric as floats. */
    xl::xout["iteration"]["5:Metric-MI"] << std::showpoint << std::fixed;
    xl::xout["iteration"]["6:Metric-RP"] << std::showpoint << std::fixed << std::setprecision( 10 );
    xl::xout["iteration"]["7:Metric-LC"] << std::showpoint << std::fixed << std::setprecision( 10 );
    xl::xout["iteration"]["8:Metric-OC"] << std::showpoint << std::fixed << std::setprecision( 10 );
    xl::xout["iteration"]["9:Metric-PC"] << std::showpoint << std::fixed << std::setprecision( 10 );

	} // end BeforeRegistration


	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationWithRigidityPenalty<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		/** Create and start a timer. */
		TimerPointer timer = TimerType::New();
		timer->StartTimer();

		/** Initialize this class with the Superclass initializer. */
		this->Superclass1::Initialize();

    /** Check stuff. */
    this->m_RigidityPenaltyTermMetric->CheckUseAndCalculationBooleans();

		/** Stop and print the timer. */
		timer->StopTimer();
		elxout << "Initialization of MattesMutualInformationWithRigidityPenalty metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationWithRigidityPenalty<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
		
		/** Get and set the number of histogram bins and spatial samples. */				
		unsigned int numberOfHistogramBins = 32;
    this->m_Configuration->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", 0 );
    this->m_Configuration->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", level );
    this->m_MattesMutualInformationMetric->SetNumberOfHistogramBins( numberOfHistogramBins );
		/** \todo guess the default numberOfSpatialSamples from the 
		 * imagesize, the numberOfParameters, and the number of bins...
		 */
	
		/** Set the DilateRigidityImages in the superclass to the one of this level. */
		this->SetDilateRigidityImages( this->m_DilateRigidityImagesVector[ level ] );

    /** Get and set the dilationRadiusMultiplier. */
		double dilationRadiusMultiplier = 1.0;
		this->GetConfiguration()->ReadParameter( dilationRadiusMultiplier, "DilationRadiusMultiplier", 0 );
    this->GetConfiguration()->ReadParameter( dilationRadiusMultiplier, "DilationRadiusMultiplier", level );
		this->SetDilationRadiusMultiplier( dilationRadiusMultiplier );

    /** Set the RigidityPenaltyWeight in the superclass to the one of this level. */
		this->SetMetricWeight( this->m_RigidityPenaltyWeightVector[ level ], 1 );

    /** Set the LinearityConditionWeight of this level. */
    double linearityConditionWeight = 1.0;
    this->m_Configuration->ReadParameter( linearityConditionWeight, "LinearityConditionWeight", 0 );
    this->m_Configuration->ReadParameter( linearityConditionWeight, "LinearityConditionWeight", level );
    this->m_RigidityPenaltyTermMetric->SetLinearityConditionWeight( linearityConditionWeight );

    /** Set the orthonormalityConditionWeight of this level. */
    double orthonormalityConditionWeight = 1.0;
    this->m_Configuration->ReadParameter( orthonormalityConditionWeight, "OrthonormalityConditionWeight", 0 );
    this->m_Configuration->ReadParameter( orthonormalityConditionWeight, "OrthonormalityConditionWeight", level );
    this->m_RigidityPenaltyTermMetric->SetOrthonormalityConditionWeight( orthonormalityConditionWeight );

    /** Set the propernessConditionWeight of this level. */
    double propernessConditionWeight = 1.0;
    this->m_Configuration->ReadParameter( propernessConditionWeight, "PropernessConditionWeight", 0 );
    this->m_Configuration->ReadParameter( propernessConditionWeight, "PropernessConditionWeight", level );
    this->m_RigidityPenaltyTermMetric->SetPropernessConditionWeight( propernessConditionWeight );
		
	} // end BeforeEachResolution
	

	/**
	 * ***************AfterEachIteration ****************************
	 */

	template <class TElastix>
		void MattesMutualInformationWithRigidityPenalty<TElastix>
		::AfterEachIteration(void)
	{
		/** Print some information. */
    xl::xout["iteration"]["5:Metric-MI"] << this->GetMetricValue( 0 );
    xl::xout["iteration"]["6:Metric-RP"] <<
      this->m_RigidityPenaltyTermMetric->GetRigidityPenaltyTermValue();
    xl::xout["iteration"]["7:Metric-LC"] <<
      this->m_RigidityPenaltyTermMetric->GetLinearityConditionValue();
    xl::xout["iteration"]["8:Metric-OC"] <<
      this->m_RigidityPenaltyTermMetric->GetOrthonormalityConditionValue();
    xl::xout["iteration"]["9:Metric-PC"] <<
      this->m_RigidityPenaltyTermMetric->GetPropernessConditionValue();

	} // end AfterEachIteration


} // end namespace elastix


#endif // end #ifndef __elxMattesMutualInformationWithRigidityPenalty_HXX__

