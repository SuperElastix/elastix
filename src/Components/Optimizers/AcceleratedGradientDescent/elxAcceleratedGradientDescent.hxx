#ifndef __elxAcceleratedGradientDescent_hxx
#define __elxAcceleratedGradientDescent_hxx

#include "elxAcceleratedGradientDescent.h"
#include <iomanip>
#include <string>
#include <vector>
#include "vnl/vnl_math.h"
#include "itkImageRandomConstIteratorWithIndex.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkAdvancedImageToImageMetric.h"
#include "itkImageRandomSamplerSparseMask.h"

namespace elastix
{
  using namespace itk;

  /**
  * ********************** Constructor ***********************
  */

  template <class TElastix>
    AcceleratedGradientDescent<TElastix>::
    AcceleratedGradientDescent()
  {
    this->m_AutomaticParameterEstimation = false;
    this->m_MaximumStepLength = 1.0;

    this->m_NumberOfGradientMeasurements = 0;
    this->m_NumberOfJacobianMeasurements = 0;
    this->m_NumberOfSamplesForExactGradient = 100000;

    this->m_UseGenericLinearMethod = true;
    
  } // Constructor


  /**
  * ***************** BeforeRegistration ***********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>::
    BeforeRegistration(void)
  {

    /** Add the target cell "stepsize" to xout["iteration"].*/
    xout["iteration"].AddTargetCell("2:Metric");
    xout["iteration"].AddTargetCell("3a:Time");
    xout["iteration"].AddTargetCell("3b:StepSize");
    xout["iteration"].AddTargetCell("4:||Gradient||");

    /** Format the metric and stepsize as floats */			
    xl::xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
    xl::xout["iteration"]["3a:StepSize"] << std::showpoint << std::fixed;
    xl::xout["iteration"]["3b:StepSize"] << std::showpoint << std::fixed;
    xl::xout["iteration"]["4:||Gradient||"] << std::showpoint << std::fixed;

    this->m_SettingsVector.clear();

  } // end BeforeRegistration


  /**
  * ***************** BeforeEachResolution ***********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::BeforeEachResolution(void)
  {
    /** Get the current resolution level.*/
    unsigned int level = static_cast<unsigned int>(
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

    /** Set the maximumNumberOfIterations.*/
    unsigned int maximumNumberOfIterations = 100;
    this->GetConfiguration()->ReadParameter( maximumNumberOfIterations,
      "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
    this->SetNumberOfIterations( maximumNumberOfIterations );

    /** Set the gain parameters */
    double a = 400.0;
    double A = 50.0;
    double alpha = 0.602;

    this->GetConfiguration()->ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter(A, "SP_A", this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0 );

    this->SetParam_a(	a );
    this->SetParam_A( A );
    this->SetParam_alpha( alpha );

    /** Set/Get whether CruzzAcceleration is desired. Default: false */
    bool usecruz = false;
    this->GetConfiguration()->ReadParameter( usecruz,
      "UseCruzAcceleration", this->GetComponentLabel(), level, 0 );
    this->SetUseCruzAcceleration( usecruz );

    if ( usecruz )
    {
      /** Set/Get the maximum of the sigmoid use by CruzAcceleration. 
      * Should be >0. Default: 1.0 */     
      double sigmoidMax = 1.0;
      this->GetConfiguration()->ReadParameter( sigmoidMax,
        "SigmoidMax", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidMax( sigmoidMax );

      /** Set/Get the maximum of the sigmoid use by CruzAcceleration. 
      * Should be <0. Default: -0.999 */     
      double sigmoidMin = -0.999;
      this->GetConfiguration()->ReadParameter( sigmoidMin,
        "SigmoidMin", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidMin( sigmoidMin );

      /** Set/Get the scaling of the sigmoid width. Large values 
      * cause a more wide sigmoid. Default: 1e-8. Should be >0. */     
      double sigmoidScale = 1e-8;
      this->GetConfiguration()->ReadParameter( sigmoidScale,
        "SigmoidScale", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidScale( sigmoidScale );

      /** Set/Get the initial time. Default: 10.0. Should be >0. */     
      double initialTime = 10.0;
      this->GetConfiguration()->ReadParameter( initialTime,
        "SigmoidInitialTime", this->GetComponentLabel(), level, 0 );
      this->SetInitialTime( initialTime );
    }

    /** Set whether automatic gain estimation is required */
    this->m_AutomaticParameterEstimation = false;
    this->GetConfiguration()->ReadParameter( this->m_AutomaticParameterEstimation,
      "AutomaticParameterEstimation", this->GetComponentLabel(), level, 0 );

    if ( this->m_AutomaticParameterEstimation )
    {
      /** Set the maximum step length: the maximum displacement of a voxel in mm  */

      /** Compute default value: */
      const unsigned int indim = this->GetElastix()->FixedDimension;
      this->m_MaximumStepLength = itk::NumericTraits<double>::max();
      for (unsigned int d = 0; d < indim; ++d )
      {
        this->m_MaximumStepLength = vnl_math_min( 
          this->m_MaximumStepLength,
          this->GetElastix()->GetFixedImage()->GetSpacing()[d] );
      }
      /** Read user setting. */
      this->GetConfiguration()->ReadParameter( this->m_MaximumStepLength,
        "MaximumStepLength", this->GetComponentLabel(), level, 0 );

      /** Read some parameters which are interesting for research only: */
      this->m_NumberOfGradientMeasurements = 0;
      this->GetConfiguration()->ReadParameter(
        this->m_NumberOfGradientMeasurements,
        "NumberOfGradientMeasurements",
        this->GetComponentLabel(), level, 0 );

      this->m_NumberOfJacobianMeasurements = 0;
      this->GetConfiguration()->ReadParameter(
        this->m_NumberOfJacobianMeasurements,
        "NumberOfJacobianMeasurements",
        this->GetComponentLabel(), level, 0 );

      this->m_NumberOfSamplesForExactGradient = 100000;
      this->GetConfiguration()->ReadParameter(
        this->m_NumberOfSamplesForExactGradient,
        "NumberOfSamplesForExactGradient ",
        this->GetComponentLabel(), level, 0 );   

      this->m_UseGenericLinearMethod = true;
       this->GetConfiguration()->ReadParameter( this->m_UseGenericLinearMethod,
         "UseGenericLinearMethod", this->GetComponentLabel(), level, 0 );

    } // end if automatic parameter estimation


  } // end BeforeEachResolution


  /**
  * ***************** AfterEachIteration *************************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::AfterEachIteration(void)
  {
    /** Print some information */
    xl::xout["iteration"]["2:Metric"]		<< this->GetValue();
    xl::xout["iteration"]["3a:Time"] << this->GetCurrentTime();
    xl::xout["iteration"]["3b:StepSize"] << this->GetLearningRate();
    xl::xout["iteration"]["4:||Gradient||"] << this->GetGradient().magnitude();


    /** Select new spatial samples for the computation of the metric */
    if ( this->GetNewSamplesEveryIteration() )
    {
      this->SelectNewSamples();
    }

  } // end AfterEachIteration


  /**
  * ***************** AfterEachResolution *************************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::AfterEachResolution(void)
  {
    /** Get the current resolution level.*/
    unsigned int level = static_cast<unsigned int>(
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

    /**
    * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }  
    */
    std::string stopcondition;


    switch( this->GetStopCondition() )
    {

    case MaximumNumberOfIterations :
      stopcondition = "Maximum number of iterations has been reached";	
      break;	

    case MetricError :
      stopcondition = "Error in metric";	
      break;	

    default:
      stopcondition = "Unknown";
      break;

    }

    /** Print the stopping condition */
    elxout << "Stopping condition: " << stopcondition << "." << std::endl;

    /** Store the used parameters, for later printing to screen */
    SettingsType settings;    
    settings.a = this->GetParam_a();
    settings.A = this->GetParam_A();
    settings.alpha = this->GetParam_alpha();
    settings.fmax = this->GetSigmoidMax();
    settings.fmin = this->GetSigmoidMin();
    settings.omega = this->GetSigmoidScale();
    this->m_SettingsVector.push_back( settings );

    /** Print settings that were used in this resolution */
    SettingsVectorType tempSettingsVector;
    tempSettingsVector.push_back( settings );
    elxout 
      << "Settings of " << this->elxGetClassName() 
      << " in resolution " << level << ":" << std::endl;
    this->PrintSettingsVector( tempSettingsVector );

  } // end AfterEachResolution


  /**
  * ******************* AfterRegistration ************************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::AfterRegistration(void)
  {
    /** Print the best metric value */

    double bestValue = this->GetValue();
    elxout
      << std::endl
      << "Final metric value  = " 
      << bestValue
      << std::endl;

    elxout
      << "Settings of " << this->elxGetClassName()
      << " for all resolutions:" << std::endl;
    this->PrintSettingsVector( this->m_SettingsVector );

  } // end AfterRegistration


  /**
  * ****************** StartOptimization *************************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::StartOptimization(void)
  {
    /** Check if the entered scales are correct and != [ 1 1 1 ...] */

    this->SetUseScales(false);
    const ScalesType & scales = this->GetScales();
    if ( scales.GetSize() == this->GetInitialPosition().GetSize() )
    {
      ScalesType unit_scales( scales.GetSize() );
      unit_scales.Fill(1.0);
      if (scales != unit_scales)
      {
        /** only then: */
        this->SetUseScales(true);
      }
    }

    this->Superclass1::StartOptimization();

  } //end StartOptimization


  /** 
  * ********************** ResumeOptimization **********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::ResumeOptimization(void)
  {
    if ( this->GetAutomaticParameterEstimation() )
    {
      this->AutomaticParameterEstimation();
    }

    this->Superclass1::ResumeOptimization();

  } //end ResumeOptimization


  /** 
  * ******************* AutomaticParameterEstimation **********************
  * Estimates some reasonable values for the parameters
  * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and SigmoidScale. 
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::AutomaticParameterEstimation( void )
  {
    /** Get the user input */
    const double delta = this->GetMaximumStepLength();

    /** Compute the jacobian terms */
    double TrC = 0.0;
    double TrCC = 0.0;
    double maxJJ = 0.0;
    double maxJCJ = 0.0;
    this->ComputeJacobianTerms(TrC, TrCC, maxJJ, maxJCJ);

    /** Measure square magnitude of exact gradient and approximation error */
    const double sigma4 = delta / vcl_sqrt( maxJJ );
    double gg = 0.0;
    double ee = 0.0;
    this->SampleGradients( this->GetScaledCurrentPosition(), sigma4, gg, ee );

    /** Determine parameter settings */
    const double sigma1 = vcl_sqrt( gg / TrC );
    const double sigma3 = vcl_sqrt( ee / TrC );

    const double alpha = 1.0;
    const double A = this->GetParam_A();
    const double a_max = A * delta / sigma1  / vcl_sqrt( maxJCJ );
    const double noisefactor = gg / ( gg + ee + 1e-14 );
    const double a = a_max * noisefactor;

    const double omega = 0.1 * sigma3 * sigma3 * vcl_sqrt( TrCC );
    const double fmax = 1.0;
    const double fmin = -0.99+ 0.98*noisefactor;
        
    /** Set parameters in superclass */
    this->SetParam_a( a );
    this->SetParam_alpha( alpha );
    this->SetSigmoidMax( fmax );
    this->SetSigmoidMin( fmin );
    this->SetSigmoidScale( omega );
   
  } // end AutomaticParameterEstimation


  /** 
  * ******************** SampleGradients **********************
  */

  /** Measure some derivatives, exact and approximated. Returns
  * the squared magnitude of the gradient and approximation error.
  * Needed for the automatic parameter estimation */
  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::SampleGradients(const ParametersType & mu0,
    double perturbationSigma, double & gg, double & ee)
  {
    typedef itk::AdvancedImageToImageMetric<
      FixedImageType, MovingImageType>                  AdvancedMetricType;
    typedef typename AdvancedMetricType::Pointer        AdvancedMetricPointer;
    typedef itk::ImageRandomSamplerBase<FixedImageType> ImageRandomSamplerType;
    typedef typename ImageRandomSamplerType::Pointer    ImageRandomSamplerPointer;
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator 
                                                        RandomGeneratorType;

    const unsigned int P = static_cast<unsigned int>( mu0.GetSize() );
    const double Pd = static_cast<double>( P );

    /** Number of gradients N to estimate the average square magnitude.
     * Use the user entered value or a default if the user specified 0.
     * N * nrofparams = 500
     * This gives a probability of ~1 that the average square magnitude
     * does not exceed twice the real expected square magnitude, or half.  */
    unsigned int numberofgradients = this->m_NumberOfGradientMeasurements;
    if ( numberofgradients == 0 )
    {
      numberofgradients = static_cast<unsigned int>( vcl_ceil( 500.0 / Pd ) );
    }

    bool stochasticgradients = this->GetNewSamplesEveryIteration();
    ImageRandomSamplerPointer sampler = 0;
    AdvancedMetricPointer advmetric = 0;
    unsigned int normalnumberofsamples = 0;
    const unsigned int allsamples = this->m_NumberOfSamplesForExactGradient;
    double dummyvalue = 0.0;
    DerivativeType approxgradient;
    DerivativeType exactgradient;
    DerivativeType diffgradient;
    double exactgg = 0.0;
    double diffgg = 0.0;
    double approxgg = 0.0; 

    /** Find the sampler; does not work for multimetric! */
    if ( stochasticgradients )
    {
      advmetric = dynamic_cast<AdvancedMetricType * >( 
        this->GetElastix()->GetElxMetricBase() );
      if (advmetric)
      {
        sampler = dynamic_cast<ImageRandomSamplerType*>( advmetric->GetImageSampler() );
        if ( (!advmetric->GetUseImageSampler()) || sampler.IsNull() )
        {
          stochasticgradients = false;
        }
        else
        {
          normalnumberofsamples = sampler->GetNumberOfSamples();
        }
      }
      else
      {
        stochasticgradients = false;
      }
    }     

    /** Compute gg for some random parameters */      
    typename RandomGeneratorType::Pointer randomgenerator = RandomGeneratorType::New();
    for ( unsigned int i = 0 ; i < numberofgradients; ++i)
    {
      /** Generate a perturbation; actually we should generate a perturbation 
      * with the same expected sqr magnitude as E||g||^2 = frofrojac 
      * The expected sqr magnitude of a N-D normal distribution N(0,I) is N,
      * so, the perturbation gain needs to be multiplied by frofrojac/Nd.  */
      ParametersType perturbation = mu0;
      for (unsigned int p = 0; p < P; ++p)
      {
        perturbation[p] += perturbationSigma *
          randomgenerator->GetNormalVariate(0.0, 1.0);
      }

      /** Select new spatial samples for the computation of the metric */
      if ( stochasticgradients )
      {
        sampler->SetNumberOfSamples( normalnumberofsamples );
        this->SelectNewSamples();
      }

      /** Get approximate derivative and its magnitude */
      this->GetScaledValueAndDerivative( perturbation, dummyvalue, approxgradient );
      approxgg += approxgradient.squared_magnitude();

      /** Get exact gradient and its magnitude */
      if ( stochasticgradients )
      {
        sampler->SetNumberOfSamples( allsamples );
        this->SelectNewSamples();
        this->GetScaledValueAndDerivative( perturbation, dummyvalue, exactgradient );
        exactgg += exactgradient.squared_magnitude();
        diffgradient = exactgradient - approxgradient;
        diffgg += diffgradient.squared_magnitude();
      }
      else
      {
        exactgg = approxgg;
        diffgg = 0.0;
      }
    } // end for
    approxgg /= numberofgradients;
    exactgg /= numberofgradients;
    diffgg /= numberofgradients;
        
    if (stochasticgradients)
    {
      /** Set back to what it was */
      sampler->SetNumberOfSamples( normalnumberofsamples );
    }    

    /** For output: */
    gg = exactgg;
    ee = diffgg;

  } // end SampleGradients


  /** 
  * ******************** ComputeJacobianTerms **********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::ComputeJacobianTerms(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {
    std::string transformName = this->GetElastix()->
      GetElxTransformBase()->GetNameOfClass();

    const std::string translationName = "TranslationTransformElastix";
    if ( transformName == translationName )
    {
      this->ComputeJacobianTermsTranslation(
        TrC, TrCC, maxJJ, maxJCJ );
    }
    else if ( this->m_UseGenericLinearMethod ) 
    {
      this->ComputeJacobianTermsGenericLinear(
        TrC, TrCC, maxJJ, maxJCJ );
    }
    else
    {
      this->ComputeJacobianTermsGenericQuadratic(
        TrC, TrCC, maxJJ, maxJCJ );
    }
    
  } // end ComputeJacobianTerms


  /** 
  * *********** ComputeJacobianTermsGenericQuadratic ************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::ComputeJacobianTermsGenericQuadratic(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {
    typedef std::vector< JacobianType >                 JacobianVectorType;

    /** Get samples */
    ImageSampleContainerPointer sampleContainer = 0;
    this->SampleFixedImageForJacobianTerms( sampleContainer );
    unsigned int nrofsamples = sampleContainer->Size();
    const double n = static_cast<double>(nrofsamples);

    /** Get the number of parameters */
    const unsigned int P = static_cast<unsigned int>( 
      this->GetScaledCurrentPosition().GetSize() );
    const double Pd = static_cast<double>( P );

    /** Get transform and set current position */
    typename TransformType::Pointer transform = this->GetRegistration()->
      GetAsITKBaseType()->GetTransform();
    transform->SetParameters( this->GetCurrentPosition() );
    const unsigned int outdim = transform->GetOutputSpaceDimension();
    const double outdimd = static_cast<double>( outdim );
    
    /** Get scales vector */
    const ScalesType & scales = this->m_ScaledCostFunction->GetScales();
    
    /** Prepare jacobian container */
    JacobianVectorType jacvec(nrofsamples);
        
    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator iter;
    typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

    /** Initialize */
    unsigned int s = 0;    

		/** Loop over image and compute jacobian. Save the jacobians in a vector. */
    for ( iter = begin; iter != end; ++iter )
		{
	    /** Read fixed coordinates and get jacobian. */
      const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
      jacvec[s] = transform->GetJacobian( point );

      /** Apply scales, if necessary */
      if ( this->GetUseScales() )
      {
        for (unsigned int p = 0; p < P; ++p)
        {
          jacvec[s].scale_column( p, 1.0/scales[p] );
        }
      }        
      ++s;
    } // end for loop over samples

    /** Compute the stuff in a double loop over the jacobians 
     * \li TrC = 1/n \sum_j ||J_j||_F^2
     * \li maxJJ = max_j [ ||J_j||_F^2 + 2\sqrt{2} || J_j J_j^T ||_F ]
     * \li maxJCJ = max_j [ 1/n \sum_i ||J_j J_i^T||_F^2 + 
     *   2\sqrt{2} 1/n || \sum_i (J_j J_i^T) (J_j J_i^T)^T ||_F ]
     * \li TrCC = 1/n^2 sum_i sum_j || J_j J_i^T ||_F^2
     */
    TrC = 0.0;
    TrCC = 0.0;
    maxJJ = 0.0;
    maxJCJ = 0.0;
    const double sqrt2 = vcl_sqrt(static_cast<double>(2.0));
    for ( unsigned int j = 0 ; j < nrofsamples; ++j)
    {      
      const JacobianType & jacj = jacvec[j];

      /** TrC = 1/n \sum_j ||J_j||_F^2 */
      const double fro2jacj = vnl_math_sqr( jacj.frobenius_norm() );
      TrC += fro2jacj / n; 

      /** Compute 1st part of JJ: ||J_j||_F^2 */
      double JJ_j = fro2jacj;

      /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F */
      JacobianType jacjjacj(outdim,outdim); // J_j J_j^T
      for( unsigned int dx = 0; dx < outdim; ++dx )
      {
        for( unsigned int dy = 0; dy < outdim; ++dy )
        {
          jacjjacj(dx,dy)=0.0;
          for (unsigned int p = 0; p < P; ++p)
          {
            jacjjacj[dx][dy] += jacj[dx][p] * jacj[dy][p];
          } // p
        } // dy
      } // dx
      JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

      /** Max_j [JJ] */
      maxJJ = vnl_math_max( maxJJ, JJ_j);

      /** Compute JCJ */
      double JCJ_j = 0.0;
      JacobianType jac4(outdim,outdim);
      jac4.Fill(0.0); // = \sum_i (J_j J_i^T) (J_j J_i^T)^T
      for ( unsigned int i = 0 ; i < nrofsamples; ++i)
      {
        const JacobianType & jaci = jacvec[i];

        JacobianType jacjjaci(outdim,outdim); // J_j J_i^T
        for( unsigned int dx = 0; dx < outdim; ++dx )
        {
          ParametersType jacjdx(jacj[dx], P, false);
          for( unsigned int dy = 0; dy < outdim; ++dy )
          {
            ParametersType jacidy(jaci[dy], outdim, false);
            jacjjaci(dx,dy)= dot_product(jacjdx, jacidy);
          } // dy
        } // dx

        const double fro2jacjjaci = vnl_math_sqr( jacjjaci.frobenius_norm() );

        /** Update TrCC: TrCC = 1/n^2 sum_i sum_j || J_j J_i^T ||_F^2 */
        TrCC += fro2jacjjaci / n / n;

        /** Update 1st part of JCJ: 1/n \sum_i ||J_j J_i^T||_F^2 */
        JCJ_j += fro2jacjjaci / n;
       
        /** Prepare for 2nd part of JCJ: Update \sum_i (J_j J_i^T) (J_j J_i^T)^T  */
        for( unsigned int dx = 0; dx < outdim; ++dx )
        {
          ParametersType jacjjacidx(jacjjaci[dx], outdim, false);
          for( unsigned int dy = 0; dy < outdim; ++dy )
          {              
            ParametersType jacjjacidy(jacjjaci[dy], outdim, false);
            jac4[dx][dy] += dot_product(jacjjacidx, jacjjacidy);
          }
        } 
        
      }  // next i

      /** Update 2nd part of JCJ: 
       * 2\sqrt{2} 1/n || \sum_i (J_j J_i^T) (J_j J_i^T)^T ||_F */
      JCJ_j += 2.0 * sqrt2 * jac4.frobenius_norm() / n;
      
      /** Max_j [JCJ]*/
      maxJCJ = vnl_math_max( maxJCJ, JCJ_j);

    } // next j
    
    /** Clean up */
    jacvec.clear();

  } // end ComputeJacobianTermsGenericQuadratic


  /** 
  * ************* ComputeJacobianTermsGenericLinear ****************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::ComputeJacobianTermsGenericLinear(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {
    typedef itk::Array2D<double>                        CovarianceMatrixType;   
    typedef CovarianceMatrixType::iterator              CovarianceMatrixIteratorType;
    typedef JacobianType::const_iterator                JacobianConstIteratorType;
    typedef vnl_vector<double>                          JacobianColumnType;

    /** Get samples */
    ImageSampleContainerPointer sampleContainer = 0;
    this->SampleFixedImageForJacobianTerms( sampleContainer );
    unsigned int nrofsamples = sampleContainer->Size();
    const double n = static_cast<double>(nrofsamples);

    /** Get the number of parameters */
    const unsigned int P = static_cast<unsigned int>( 
      this->GetScaledCurrentPosition().GetSize() );
    const double Pd = static_cast<double>( P );

    /** Get transform and set current position */
    typename TransformType::Pointer transform = this->GetRegistration()->
      GetAsITKBaseType()->GetTransform();
    transform->SetParameters( this->GetCurrentPosition() );
    const unsigned int outdim = transform->GetOutputSpaceDimension();
    const double outdimd = static_cast<double>( outdim );

    /** Get scales vector */
    const ScalesType & scales = this->m_ScaledCostFunction->GetScales();
                
    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator iter;
    typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

    /** Initialize */
    CovarianceMatrixType cov(P,P);
    cov.Fill(0.0);    
    
		/** Loop over image and compute jacobian. Possibly apply scaling.
     * Compute C = 1/n \sum_i J_i^T J_i */
    std::vector<JacobianConstIteratorType> jacit(outdim);
    CovarianceMatrixIteratorType covit;
    for ( iter = begin; iter != end; ++iter )
		{
	    /** Read fixed coordinates and get jacobian. 
       * \todo: extend for sparse jacobians */
      const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
      const JacobianType & jac = transform->GetJacobian( point );   
     
      /** Update covariance matrix */
      covit = cov.begin();
      for ( unsigned int p = 0; p < P; ++p )
      {        
        const JacobianColumnType jaccolp = jac.get_column(p);
        /** Initialize iterators at first column of jacobian */
        for ( unsigned int d = 0; d < outdim; ++d)
        {
          jacit[d] = jac.begin() + d * P;
        }
        for ( unsigned int q = 0; q < P; ++q )
        {          
          for ( unsigned int d = 0; d < outdim; ++d)
          {
            *covit += jaccolp[d] * (*jacit[d]) / n;
            ++jacit[d];
          }          
          ++covit;
        } // q
      } // p       
        
    } // end computation of covariance matrix

    /** Apply scales. */
    if ( this->GetUseScales() )
    {
      for (unsigned int p = 0; p < P; ++p)
      {
        cov.scale_column( p, 1.0/scales[p] );
        cov.scale_row( p, 1.0/scales[p] );
      }
    }

    /** Compute TrC = trace(C) */
    for (unsigned int p = 0; p < P; ++p)
    {
      TrC += cov[p][p];
    }

    /** Compute TrCC = ||C||_F^2 */
    TrCC = vnl_math_sqr( cov.frobenius_norm() );

    /** Compute maxJJ and maxJCJ
     * \li maxJJ = max_j [ ||J_j||_F^2 + 2\sqrt{2} || J_j J_j^T ||_F ]
     * \li maxJCJ = max_j [ Tr( J_j C J_j^T ) + 2\sqrt{2} || J_j C J_j^T ||_F ]
     */
    maxJJ = 0.0;
    maxJCJ = 0.0;    
    const double sqrt2 = vcl_sqrt(static_cast<double>(2.0));
    JacobianType jacj;
    for ( iter = begin; iter != end; ++iter )
		{
	    /** Read fixed coordinates and get jacobian. */      
      const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
      jacj = transform->GetJacobian( point );

      /** Apply scales, if necessary */
      if ( this->GetUseScales() )
      {
        for (unsigned int p = 0; p < P; ++p)
        {
          jacj.scale_column( p, 1.0/scales[p] );
        }
      } 
  
      /** Compute 1st part of JJ: ||J_j||_F^2 */
      double JJ_j = vnl_math_sqr( jacj.frobenius_norm() );

      /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F */
      JacobianType jacjjacj(outdim,outdim); // J_j J_j^T
      for( unsigned int dx = 0; dx < outdim; ++dx )
      {
        for( unsigned int dy = 0; dy < outdim; ++dy )
        {
          jacjjacj(dx,dy)=0.0;
          for (unsigned int p = 0; p < P; ++p)
          {
            jacjjacj[dx][dy] += jacj[dx][p] * jacj[dy][p];
          } // p
        } // dy
      } // dx
      JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

      /** Max_j [JJ] */
      maxJJ = vnl_math_max( maxJJ, JJ_j);

      /** Compute JCJ */
      double JCJ_j = 0.0;
      
      /** J_j C */
      JacobianType jacjC(outdim, P);
      jacjC = jacj * cov;
     
      /** J_j C J_j^T */
      JacobianType jacjCjacj(outdim, outdim);
      for( unsigned int dx = 0; dx < outdim; ++dx )
      {
        ParametersType jacjCdx(jacjC[dx], P, false);
        for( unsigned int dy = 0; dy < outdim; ++dy )
        {
          ParametersType jacjdy(jacj[dy], outdim, false);
          jacjCjacj(dx,dy) = dot_product(jacjCdx, jacjdy);
        } // dy
      } // dx
      
      /** Compute 1st part of JCJ: Tr( J_j C J_j^T ) */
      for (unsigned int d = 0; d < outdim; ++d)
      {
        JCJ_j += jacjCjacj[d][d];
      }

      /** Compute 2nd part of JCJ: 2\sqrt{2} || J_j C J_j^T ||_F */
      JCJ_j += 2.0 * sqrt2 * jacjCjacj.frobenius_norm();
      
      /** Max_j [JCJ]*/
      maxJCJ = vnl_math_max( maxJCJ, JCJ_j);

    } // next sample from sample container     

  } // end ComputeJacobianTermsGenericLinear


  /** 
  * ***************** ComputeJacobianTermsAffine **********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::ComputeJacobianTermsAffine(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {} // end ComputeJacobianTermsAffine


  /** 
  * ************* ComputeJacobianTermsTranslation ********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::ComputeJacobianTermsTranslation(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {
    /** Get the number of parameters */
    const unsigned int P = static_cast<unsigned int>( 
      this->GetScaledCurrentPosition().GetSize() );
    const double Pd = static_cast<double>( P );

    const double sqrt2 = vcl_sqrt(static_cast<double>(2.0));

    /** For translation transforms the Jacobian dT/dmu equals I
     * at every voxel. The Jacobian terms are simplified in this case: */
    TrC = Pd;
    TrCC = Pd;
    maxJJ = Pd + 2.0 * sqrt2 * vcl_sqrt(Pd);
    maxJCJ = maxJJ;

  } // end ComputeJacobianTermsTranslation


  /** 
  * **************** ComputeJacobianTermsBSpline **********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::ComputeJacobianTermsBSpline(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {} // end ComputeJacobianTermsBSpline


  /** 
  * **************** SampleFixedImageForJacobianTerms *******************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::SampleFixedImageForJacobianTerms(
    ImageSampleContainerPointer & sampleContainer )
  {
    /** Get fixed image and region */
    typename FixedImageType::ConstPointer fixedImage = this->GetElastix()->
      GetElxMetricBase()->GetAsITKBaseType()->GetFixedImage();
    FixedImageRegionType fixedRegion = this->GetElastix()->
      GetElxMetricBase()->GetAsITKBaseType()->GetFixedImageRegion();
    const double fixdimd = static_cast<double>( fixedRegion.GetImageDimension() );

    /** Set up grid sampler */
    ImageSamplerPointer sampler = ImageSamplerType::New();
    sampler->SetInput( fixedImage );
    sampler->SetInputImageRegion( fixedRegion );
    sampler->SetMask( this->GetElastix()->GetElxMetricBase()->
      GetAsITKBaseType()->GetFixedImageMask() );
    
    /** Determine grid spacing of sampler for each dimension
     * gridspacing = round[
     *  (nrofpixelsinfixedregion / desirednumberofjacobianmeasurements)^(1/D) ]
     * and at least 1. 
     * Note that the actual number of samples may be lower, due to masks */

    unsigned int nrofsamples = 200;
    
    /** Check user input; an input of 0 means that the default is used. */
    if ( this->m_NumberOfJacobianMeasurements != 0 )
    {
      nrofsamples = this->m_NumberOfJacobianMeasurements;
    }

    /** Compute the grid spacing */
    const double fraction = 
      static_cast<double>( fixedRegion.GetNumberOfPixels() ) /
      static_cast<double>( nrofsamples );
    int gridspacing = static_cast<int>( 
      vnl_math_rnd( vcl_pow(fraction, 1.0/fixdimd) )   );
    gridspacing = vnl_math_max( 1, gridspacing );
    typename ImageSamplerType::SampleGridSpacingType gridspacings;
    gridspacings.Fill( gridspacing );
    sampler->SetSampleGridSpacing( gridspacings );

    /** get samples and check the actually obtained number of samples */
    sampler->Update();
    sampleContainer = sampler->GetOutput();
    nrofsamples = sampleContainer->Size();
    if ( nrofsamples == 0 )
    {
      itkExceptionMacro(
        << "No valid voxels found to estimate the AcceleratedGradientDescent parameters." );
    }
    
  } // end SampleFixedImageForJacobianTerms


  /** 
  * **************** PrintSettingsVector **********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::PrintSettingsVector( const SettingsVectorType & settings ) const
  {
    const unsigned long nrofres = settings.size();

    /** Print to log file */
    elxout << "( SP_a " ;
    for (unsigned int i = 0; i < nrofres; ++i)
    {
      elxout << settings[i].a << " "; 
    }
    elxout << ")\n" ;

    elxout << "( SP_A " ;
    for (unsigned int i = 0; i < nrofres; ++i)
    {
      elxout << settings[i].A << " "; 
    }
    elxout << ")\n" ;
    
    elxout << "( SP_alpha " ;
    for (unsigned int i = 0; i < nrofres; ++i)
    {
      elxout << settings[i].alpha << " "; 
    }
    elxout << ")\n" ;

    elxout << "( SigmoidMax " ;
    for (unsigned int i = 0; i < nrofres; ++i)
    {
      elxout << settings[i].fmax << " "; 
    }
    elxout << ")\n" ;

    elxout << "( SigmoidMin " ;
    for (unsigned int i = 0; i < nrofres; ++i)
    {
      elxout << settings[i].fmin << " "; 
    }
    elxout << ")\n" ;
    
    elxout << "( SigmoidScale " ;
    for (unsigned int i = 0; i < nrofres; ++i)
    {
      elxout << settings[i].omega << " "; 
    }
    elxout << ")\n" ;
    
    elxout << std::endl;

  } // end PrintSettingsVector

} // end namespace elastix

#endif // end #ifndef __elxAcceleratedGradientDescent_hxx

