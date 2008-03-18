#ifndef __elxMattesMutualInformationWithRigidityPenalty_HXX__
#define __elxMattesMutualInformationWithRigidityPenalty_HXX__

#include "elxMattesMutualInformationWithRigidityPenalty.h"

#include "itkHardLimiterFunction.h"
#include "itkExponentialLimiterFunction.h"
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

    this->m_MattesMutualInformationMetric
      ->SetUseDerivative( true );

	} // end Constructor()


	/**
	 * ********************** BeforeRegistration *********************
	 */
	
	template <class TElastix>
		void MattesMutualInformationWithRigidityPenalty<TElastix>
		::BeforeRegistration( void )
	{
		/** Get and set the useFixedRigidityImage and read the FixedRigidityImage if wanted. */
		bool useFixedRigidityImage = true;
		this->GetConfiguration()->ReadParameter( useFixedRigidityImage,
      "UseFixedRigidityImage", this->GetComponentLabel(), 0, -1 );
		if ( useFixedRigidityImage )
		{
			/** Use the FixedRigidityImage. */
			this->SetUseFixedRigidityImage( true );

			/** Read the fixed rigidity image and set it in the right class. */
			std::string fixedRigidityImageName = "";
			this->GetConfiguration()->ReadParameter( fixedRigidityImageName,
        "FixedRigidityImageName", this->GetComponentLabel(), 0, -1 );

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
					excp.SetLocation( "MattesMutualInformationWithRigidityPenalty - BeforeRegistration()" );
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
		bool useMovingRigidityImage = true;
		this->GetConfiguration()->ReadParameter( useMovingRigidityImage,
      "UseMovingRigidityImage", this->GetComponentLabel(), 0, -1 );
		if ( useMovingRigidityImage )
		{
			/** Use the movingRigidityImage. */
			this->SetUseMovingRigidityImage( true );
			
			/** Read the moving rigidity image and set it in the right class. */
			std::string movingRigidityImageName = "";
			this->GetConfiguration()->ReadParameter( movingRigidityImageName,
        "MovingRigidityImageName", this->GetComponentLabel(), 0, -1 );
      
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
					excp.SetLocation( "MattesMutualInformationWithRigidityPenalty - BeforeRegistration()" );
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
		if ( !useFixedRigidityImage && !useMovingRigidityImage )
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

	} // end BeforeRegistration()


	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationWithRigidityPenalty<TElastix>
		::Initialize( void ) throw (ExceptionObject)
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

	} // end Initialize()

	
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
		
    /**
     *  Set options for the Mattes mutual information metric.
     */

    /** Get and set the number of histogram bins. */
		unsigned int numberOfHistogramBins = 32;
    this->GetConfiguration()->ReadParameter( numberOfHistogramBins,
      "NumberOfHistogramBins", this->GetComponentLabel(), level, 0 );
		this->m_MattesMutualInformationMetric
      ->SetNumberOfFixedHistogramBins( numberOfHistogramBins );
    this->m_MattesMutualInformationMetric
      ->SetNumberOfMovingHistogramBins( numberOfHistogramBins );

    unsigned int numberOfFixedHistogramBins = numberOfHistogramBins;
    unsigned int numberOfMovingHistogramBins = numberOfHistogramBins;
    this->GetConfiguration()->ReadParameter( numberOfFixedHistogramBins,
      "NumberOfFixedHistogramBins", this->GetComponentLabel(), level, 0 );
		this->GetConfiguration()->ReadParameter( numberOfMovingHistogramBins,
      "NumberOfMovingHistogramBins", this->GetComponentLabel(), level, 0 );
		this->m_MattesMutualInformationMetric
      ->SetNumberOfFixedHistogramBins( numberOfFixedHistogramBins );
    this->m_MattesMutualInformationMetric
      ->SetNumberOfMovingHistogramBins( numberOfMovingHistogramBins );
		
    /** Set limiters. */
    typedef HardLimiterFunction< RealType, FixedImageDimension > FixedLimiterType;
    typedef ExponentialLimiterFunction< RealType, MovingImageDimension > MovingLimiterType;
    this->m_MattesMutualInformationMetric
      ->SetFixedImageLimiter( FixedLimiterType::New() );
    this->m_MattesMutualInformationMetric
      ->SetMovingImageLimiter( MovingLimiterType::New() );
    
    /** Get and set the limit range ratios. */
		double fixedLimitRangeRatio = 0.01;
    double movingLimitRangeRatio = 0.01;
    this->GetConfiguration()->ReadParameter( fixedLimitRangeRatio,
      "FixedLimitRangeRatio", this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter( movingLimitRangeRatio,
      "MovingLimitRangeRatio", this->GetComponentLabel(), level, 0 );
		this->m_MattesMutualInformationMetric
      ->SetFixedLimitRangeRatio( fixedLimitRangeRatio );
    this->m_MattesMutualInformationMetric
      ->SetMovingLimitRangeRatio( movingLimitRangeRatio );

    /** Set B-spline parzen kernel orders. */
    unsigned int fixedKernelBSplineOrder = 0;
    unsigned int movingKernelBSplineOrder = 3;
    this->GetConfiguration()->ReadParameter( fixedKernelBSplineOrder,
      "FixedKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter( movingKernelBSplineOrder, 
      "MovingKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
		this->m_MattesMutualInformationMetric
      ->SetFixedKernelBSplineOrder( fixedKernelBSplineOrder );
    this->m_MattesMutualInformationMetric
      ->SetMovingKernelBSplineOrder( movingKernelBSplineOrder );

    /**
     *  Set options for the rigidity penalty term metric.
     */

    /** Get and set the dilateRigidityImages. */
    bool dilateRigidityImages = true;
    this->GetConfiguration()->ReadParameter( dilateRigidityImages, 
      "DilateRigidityImages", this->GetComponentLabel(), level, 0 );
    this->SetDilateRigidityImages( dilateRigidityImages );

    /** Get and set the dilationRadiusMultiplier. */
		double dilationRadiusMultiplier = 1.0;
    this->GetConfiguration()->ReadParameter( dilationRadiusMultiplier,
      "DilationRadiusMultiplier", this->GetComponentLabel(), level, 0 );
		this->SetDilationRadiusMultiplier( dilationRadiusMultiplier );

    /** Get and set the usage of the linearity condition part. */ 
    bool useLinearityCondition = true;
    this->GetConfiguration()->ReadParameter( useLinearityCondition,
      "UseLinearityCondition", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetUseLinearityCondition( useLinearityCondition );

    /** Get and set the usage of the orthonormality condition part. */
    bool useOrthonormalityCondition = true;
    this->GetConfiguration()->ReadParameter( useOrthonormalityCondition,
      "UseOrthonormalityCondition", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetUseOrthonormalityCondition( useOrthonormalityCondition );

    /** Set the usage of the properness condition part. */
    bool usePropernessCondition = true;
    this->GetConfiguration()->ReadParameter( usePropernessCondition,
      "UsePropernessCondition", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetUsePropernessCondition( usePropernessCondition );

    /** Set the calculation of the linearity condition part. */
    bool calculateLinearityCondition = true;
    this->GetConfiguration()->ReadParameter( calculateLinearityCondition,
      "CalculateLinearityCondition", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetCalculateLinearityCondition( calculateLinearityCondition );

    /** Set the calculation of the orthonormality condition part. */
    bool calculateOrthonormalityCondition = true;
    this->GetConfiguration()->ReadParameter( calculateOrthonormalityCondition,
      "CalculateOrthonormalityCondition", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetCalculateOrthonormalityCondition( calculateOrthonormalityCondition );

    /** Set the calculation of the properness condition part. */
    bool calculatePropernessCondition = true;
    this->GetConfiguration()->ReadParameter( calculatePropernessCondition,
      "CalculatePropernessCondition", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetCalculatePropernessCondition( calculatePropernessCondition );

    /** Get and set the RigidityPenaltyWeight. */
    double rigidityPenaltyWeight = 1.0;
    this->m_Configuration->ReadParameter( rigidityPenaltyWeight,
        "RigidityPenaltyWeight", this->GetComponentLabel(), level, 0 );
		this->SetMetricWeight( rigidityPenaltyWeight, 1 );

    /** Set the LinearityConditionWeight of this level. */
    double linearityConditionWeight = 1.0;
    this->m_Configuration->ReadParameter( linearityConditionWeight,
      "LinearityConditionWeight", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetLinearityConditionWeight( linearityConditionWeight );

    /** Set the orthonormalityConditionWeight of this level. */
    double orthonormalityConditionWeight = 1.0;
    this->m_Configuration->ReadParameter( orthonormalityConditionWeight,
      "OrthonormalityConditionWeight", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetOrthonormalityConditionWeight( orthonormalityConditionWeight );

    /** Set the propernessConditionWeight of this level. */
    double propernessConditionWeight = 1.0;
    this->m_Configuration->ReadParameter( propernessConditionWeight,
      "PropernessConditionWeight", this->GetComponentLabel(), level, 0 );
    this->m_RigidityPenaltyTermMetric->SetPropernessConditionWeight( propernessConditionWeight );
		
	} // end BeforeEachResolution()
	

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

	} // end AfterEachIteration()


} // end namespace elastix


#endif // end #ifndef __elxMattesMutualInformationWithRigidityPenalty_HXX__

