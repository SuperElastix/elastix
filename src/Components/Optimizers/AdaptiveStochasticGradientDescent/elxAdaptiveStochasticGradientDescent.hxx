#ifndef __elxAdaptiveStochasticGradientDescent_hxx
#define __elxAdaptiveStochasticGradientDescent_hxx

#include "elxAdaptiveStochasticGradientDescent.h"
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include "vnl/vnl_math.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "vnl/vnl_matlab_filewrite.h"
#include "itkImageRandomConstIteratorWithIndex.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkAdvancedImageToImageMetric.h"
#include "itkImageRandomCoordinateSampler.h"


namespace elastix
{
  using namespace itk;

  /**
  * ********************** Constructor ***********************
  */

  template <class TElastix>
    AdaptiveStochasticGradientDescent<TElastix>::
    AdaptiveStochasticGradientDescent()
  {
    this->m_AutomaticParameterEstimation = false;
    this->m_MaximumStepLength = 1.0;

    this->m_NumberOfGradientMeasurements = 0;
    this->m_NumberOfJacobianMeasurements = 0;
    this->m_NumberOfSamplesForExactGradient = 100000;
        
    this->m_UseMaximumLikelihoodMethod = false;
    this->m_SaveCovarianceMatrix = false;

    this->m_BSplineTransform = 0;
    this->m_BSplineCombinationTransform = 0;
    this->m_NumBSplineParametersPerDim = 0;
    this->m_NumBSplineWeights = 0;
    this->m_NumberOfParameters = 0;
    this->m_TransformIsBSpline = false;
    this->m_TransformIsBSplineCombination = false;

  } // Constructor


  /**
  * ***************** BeforeRegistration ***********************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>::
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
    void AdaptiveStochasticGradientDescent<TElastix>
    ::BeforeEachResolution(void)
  {
    /** Get the current resolution level. */
    unsigned int level = static_cast<unsigned int>(
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

    const unsigned int P = this->GetElastix()->GetElxTransformBase()->
      GetAsITKBaseType()->GetNumberOfParameters();
    const double Pd = static_cast<double>( P );

    /** Set the maximumNumberOfIterations. */
    unsigned int maximumNumberOfIterations = 500;
    this->GetConfiguration()->ReadParameter( maximumNumberOfIterations,
      "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
    this->SetNumberOfIterations( maximumNumberOfIterations );

    /** Set the gain parameter A. */
    double A = 20.0;
    this->GetConfiguration()->ReadParameter(A, "SP_A", this->GetComponentLabel(), level, 0 );
    this->SetParam_A( A );
    
    /** Set/Get the initial time. Default: 0.0. Should be >=0. */     
    double initialTime = 0.0;
    this->GetConfiguration()->ReadParameter( initialTime,
      "SigmoidInitialTime", this->GetComponentLabel(), level, 0 );
    this->SetInitialTime( initialTime );

    /** Set/Get whether the adaptive step size mechanism is desired. Default: true 
     * NB: the setting is turned of in case of UseRandomSampleRegion=true. 
     * Depecrated alias UseCruzAcceleration is also still supported. */
    bool useAdaptiveStepSizes = true;
    this->GetConfiguration()->ReadParameter( useAdaptiveStepSizes,
      "UseCruzAcceleration", this->GetComponentLabel(), level, 0, true );
    this->GetConfiguration()->ReadParameter( useAdaptiveStepSizes,
      "UseAdaptiveStepSizes", this->GetComponentLabel(), level, 0 );
    this->SetUseAdaptiveStepSizes( useAdaptiveStepSizes );
 
    /** Set whether automatic gain estimation is required; default: true */
    this->m_AutomaticParameterEstimation = true;
    this->GetConfiguration()->ReadParameter( this->m_AutomaticParameterEstimation,
      "AutomaticParameterEstimation", this->GetComponentLabel(), level, 0 );

    if ( this->m_AutomaticParameterEstimation )
    {
      /** Set the maximum step length: the maximum displacement of a voxel in mm.
       * Compute default value: mean spacing of fixed and moving image */
      const unsigned int fixdim = this->GetElastix()->FixedDimension;
      const unsigned int movdim = this->GetElastix()->MovingDimension;
      double sum = 0.0;      
      for (unsigned int d = 0; d < fixdim; ++d )
      {
        sum += this->GetElastix()->GetFixedImage()->GetSpacing()[d];
      }
      for (unsigned int d = 0; d < movdim; ++d )
      {
        sum += this->GetElastix()->GetMovingImage()->GetSpacing()[d];
      }
      this->m_MaximumStepLength = sum / static_cast<double>( fixdim + movdim );
      /** Read user setting */
      this->GetConfiguration()->ReadParameter( this->m_MaximumStepLength,
        "MaximumStepLength", this->GetComponentLabel(), level, 0 );

      /** Setting: use maximum likelihood method */
      this->m_UseMaximumLikelihoodMethod = false;
      this->GetConfiguration()->ReadParameter( 
        this->m_UseMaximumLikelihoodMethod,
        "UseMaximumLikelihoodMethod",
        this->GetComponentLabel(), level, 0 );
  
      /** Setting: save .mat file with covariance matrix, sigma1, and sigma3 if true 
       * \todo: does not seem to work on linux 64bit. linux 32bit i did not test. */
      this->m_SaveCovarianceMatrix = false;
      this->GetConfiguration()->ReadParameter( 
        this->m_SaveCovarianceMatrix, "SaveCovarianceMatrix",
        this->GetComponentLabel(), level, 0 );

      /** Number of gradients N to estimate the average square magnitudes
       * of the exact gradient and the approximation error.
       * Use the following default, if nothing is specified by the user:
       * N = max( 2, min(5, 500 / nrofparams) );
       * This gives a probability of ~1 that the average square magnitude
       * does not exceed twice the real expected square magnitude, or half.  
       * The maximum value N=5 seems to be sufficient in practice. */
      const unsigned int minNrOfGradients = 2;
      const unsigned int maxNrOfGradients = 5;
      const unsigned int estimatedNrOfGradients = 
        static_cast<unsigned int>( vcl_ceil( 500.0 / Pd ) );
      this->m_NumberOfGradientMeasurements = vnl_math_max( 
        minNrOfGradients, vnl_math_min(maxNrOfGradients, estimatedNrOfGradients) );
      this->GetConfiguration()->ReadParameter(
        this->m_NumberOfGradientMeasurements,
        "NumberOfGradientMeasurements",
        this->GetComponentLabel(), level, 0 );

      /** Set the number of jacobian measurements M. 
       * By default, if nothing specified by the user, M is determined as:
       * M = max( 1000, nrofparams*3 );
       * This is a rather crude rule of thumb, which seems to work in practice. */
      this->m_NumberOfJacobianMeasurements = vnl_math_max( 
        static_cast<unsigned int>(1000), static_cast<unsigned int>(P*3) );
      this->GetConfiguration()->ReadParameter(
        this->m_NumberOfJacobianMeasurements,
        "NumberOfJacobianMeasurements",
        this->GetComponentLabel(), level, 0 );

      /** Set the number of image samples used to compute the 'exact' gradient.
       * By default, if nothing supplied by the user, 100000. This works in general.
       * If the image is smaller, the number of samples is automatically reduced later. */
      this->m_NumberOfSamplesForExactGradient = 100000;
      this->GetConfiguration()->ReadParameter(
        this->m_NumberOfSamplesForExactGradient,
        "NumberOfSamplesForExactGradient ",
        this->GetComponentLabel(), level, 0 );   

    } // end if automatic parameter estimation
    else
    {
      /** If no automatic parameter estimation is used, a and alpha also need to be specified */
      double a = 400.0; // arbitrary guess
      double alpha = 0.602;
      this->GetConfiguration()->ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0 );    
      this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0 );
      this->SetParam_a(	a );
      this->SetParam_alpha( alpha );

      /** Set/Get the maximum of the sigmoid. 
      * Should be >0. Default: 1.0 */     
      double sigmoidMax = 1.0;
      this->GetConfiguration()->ReadParameter( sigmoidMax,
        "SigmoidMax", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidMax( sigmoidMax );

      /** Set/Get the minimum of the sigmoid. 
      * Should be <0. Default: -0.8 */     
      double sigmoidMin = -0.8;
      this->GetConfiguration()->ReadParameter( sigmoidMin,
        "SigmoidMin", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidMin( sigmoidMin );

      /** Set/Get the scaling of the sigmoid width. Large values 
      * cause a more wide sigmoid. Default: 1e-8. Should be >0. */     
      double sigmoidScale = 1e-8;
      this->GetConfiguration()->ReadParameter( sigmoidScale,
        "SigmoidScale", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidScale( sigmoidScale );
    } // end else: no automatic parameter estimation

  } // end BeforeEachResolution


  /**
  * ***************** AfterEachIteration *************************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
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
    void AdaptiveStochasticGradientDescent<TElastix>
    ::AfterEachResolution(void)
  {
    /** Get the current resolution level.*/
    unsigned int level = static_cast<unsigned int>(
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

    /**
    * typedef enum {
    *   MaximumNumberOfIterations,
    *   MetricError,
    *   MinimumStepSize } StopConditionType;
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

    case MinimumStepSize :
      stopcondition = "The minimum step length has been reached";	
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
    void AdaptiveStochasticGradientDescent<TElastix>
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
    void AdaptiveStochasticGradientDescent<TElastix>
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
    void AdaptiveStochasticGradientDescent<TElastix>
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
    void AdaptiveStochasticGradientDescent<TElastix>
    ::AutomaticParameterEstimation( void )
  {
    const unsigned int P = static_cast<unsigned int>( 
      this->GetScaledCurrentPosition().GetSize() );
    const double Pd = static_cast<double>( P );

    /** Get the user input */
    const double delta = this->GetMaximumStepLength();

    /** Compute the jacobian terms */
    double TrC = 0.0;
    double TrCC = 0.0;
    double maxJJ = 0.0;
    double maxJCJ = 0.0;
    this->m_CovarianceMatrix.SetSize(0,0);
    this->ComputeJacobianTerms(TrC, TrCC, maxJJ, maxJCJ);

    /** Measure square magnitude of exact gradient and approximation error */
    const double sigma4factor = 1.0; 
    const double sigma4 = sigma4factor * delta / vcl_sqrt( maxJJ );
    double gg = 0.0;
    double ee = 0.0;
    bool maxlik = 
      this->SampleGradients( this->GetScaledCurrentPosition(), sigma4, gg, ee );

    /** Determine parameter settings */
    double sigma1;
    double sigma3;
    if ( maxlik )
    {
      /** maximum likelihood estimator of sigma: 
      * gg = 1/N sum_n g_n^T C^{-1} g_n 
      * sigma1 = gg / P */
      sigma1 = vcl_sqrt( gg / Pd );
      sigma3 = vcl_sqrt( ee / Pd );      
    }
    else
    {
      /** estimate of sigma such that empirical norm^2 equals theoretical:
      * gg = 1/N sum_n g_n' g_n
      * sigma = gg / TrC */
      sigma1 = vcl_sqrt( gg / TrC );
      sigma3 = vcl_sqrt( ee / TrC );
    }

    /** Save covariance matrix if desired */
    this->SaveCovarianceMatrix( sigma1, sigma3, this->m_CovarianceMatrix );

    /** Clean up */
    this->m_CovarianceMatrix.SetSize(0,0);

    const double alpha = 1.0;
    const double A = this->GetParam_A();
    const double a_max = A * delta / sigma1  / vcl_sqrt( maxJCJ );
    const double noisefactor = sigma1*sigma1 / ( sigma1*sigma1 + sigma3*sigma3 + 1e-14 );
    const double a = a_max * noisefactor;

    const double omega = vnl_math_max( 1e-14, 0.1 * sigma3 * sigma3 * vcl_sqrt( TrCC ) );
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
    bool AdaptiveStochasticGradientDescent<TElastix>
    ::SampleGradients(const ParametersType & mu0,
    double perturbationSigma, double & gg, double & ee)
  {
    typedef itk::AdvancedImageToImageMetric<
      FixedImageType, MovingImageType>                  AdvancedMetricType;
    typedef typename AdvancedMetricType::Pointer        AdvancedMetricPointer;
    typedef itk::ImageRandomSamplerBase<FixedImageType> ImageRandomSamplerType;
    typedef typename ImageRandomSamplerType::Pointer    ImageRandomSamplerPointer;
    typedef 
      itk::ImageRandomCoordinateSampler<FixedImageType> ImageRandomCoordinateSamplerType;
    typedef typename 
      ImageRandomCoordinateSamplerType::Pointer         ImageRandomCoordinateSamplerPointer;
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator 
      RandomGeneratorType;
    typedef vnl_symmetric_eigensystem<double>           eigType;

    /** Some shortcuts */
    const unsigned int P = static_cast<unsigned int>( mu0.GetSize() );
    const double Pd = static_cast<double>( P );
    CovarianceMatrixType & cov = this->m_CovarianceMatrix;

    /** Prepare for maximum likelihood estimation of sigmas.
     * In that case we need a matrix eigendecomposition */
    eigType * eig = 0;
    bool maxlik = false;
    unsigned int rank = P;
    if ( (cov.size() != 0) && this->m_UseMaximumLikelihoodMethod )
    {
      /** Remember that we are using the maximum likelihood method */
      maxlik = true;
            
      /** Do an eigendecomposition of the covariance matrix C 
       * and compute D^{-1/2}
       * This result will be used to compute g^T C^{-1} g, using
       * g^T C^{-1} g = x^T x, with x = D^{-1/2} V^T g */
      eig = new eigType( cov );
      for ( unsigned int i = 0; i < P; ++i )
      {
        const double tmp = eig->D(i);
        if ( tmp > 1e-16)
        {
          eig->D(i) = 1.0 / vcl_sqrt(tmp);
        }
        else
        {
          eig->D(i) = 0.0;
          --rank;
        }
      }
      /** Print rank */
      elxout << "Rank of covariance matrix is: " << rank << std::endl;
    }

    bool stochasticgradients = this->GetNewSamplesEveryIteration();
    ImageRandomSamplerPointer randomsampler = 0;
    ImageRandomCoordinateSamplerPointer randomCoordinateSampler = 0;
    bool useRandomSampleRegion = false;
    AdvancedMetricPointer advmetric = 0;
    double dummyvalue = 0.0;
    DerivativeType approxgradient;
    DerivativeType exactgradient;
    DerivativeType diffgradient;
    DerivativeType solveroutput;
    double exactgg = 0.0;
    double diffgg = 0.0;
    double approxgg = 0.0; 

    /** Find the sampler; in case of multimetric, uses only the first metric! */
    if ( stochasticgradients )
    {
      /** Check if it is possible, and get pointers to advmetric and randomsampler. */
      stochasticgradients = false;
      advmetric = dynamic_cast<AdvancedMetricType * >( 
        this->GetElastix()->GetElxMetricBase() );
      if (advmetric)
      {
        randomsampler = dynamic_cast<ImageRandomSamplerType*>( advmetric->GetImageSampler() );
        if ( advmetric->GetUseImageSampler() && randomsampler.IsNotNull() )
        {
          /** The metric has a sampler and the user set new samples every iteration: */
          stochasticgradients = true;

          /** If the sampler is a randomCoordinateSampler set the UseRandomSampleRegion
           * property to false temporarily. It disturbs the parameter estimation.
           * At the end of this function the original setting is set back. 
           * Also, the AdaptiveStepSize mechanism is turned off.
           * \todo Extend ASGD to really take into account random region sampling. */
          randomCoordinateSampler = dynamic_cast<ImageRandomCoordinateSamplerType *>(
            advmetric->GetImageSampler() );
          if ( randomCoordinateSampler.IsNotNull() )
          {
            useRandomSampleRegion = randomCoordinateSampler->GetUseRandomSampleRegion();
            if (useRandomSampleRegion)
            {
              this->SetUseAdaptiveStepSizes( false );
            }
            randomCoordinateSampler->SetUseRandomSampleRegion( false );
          }          
        } // end if random sampler
      } // end if advmetric
    } // end if stochasticgradients

    /** Set up the grid samper for the "exact" gradients */
    ImageSamplerPointer gridsampler = 0;
    if (stochasticgradients)
    {
      /** Copy settings from the random sampler and update */
      gridsampler = ImageSamplerType::New();
      gridsampler->SetInput( randomsampler->GetInput() );
      gridsampler->SetInputImageRegion( randomsampler->GetInputImageRegion() );
      gridsampler->SetMask( randomsampler->GetMask() );
      gridsampler->SetNumberOfSamples( this->m_NumberOfSamplesForExactGradient );
      gridsampler->Update();
    }

    /** Prepare for progress printing */
    ProgressCommandPointer progressObserver = ProgressCommandType::New();
    progressObserver->SetUpdateFrequency( 
      this->m_NumberOfGradientMeasurements, this->m_NumberOfGradientMeasurements );
    progressObserver->SetStartString( "  Progress: " );
    elxout << "Sampling gradients for " << this->elxGetClassName() 
      << " configuration... " << std::endl;

    /** Compute gg for some random parameters */      
    typename RandomGeneratorType::Pointer randomgenerator = RandomGeneratorType::New();
    for ( unsigned int i = 0 ; i < this->m_NumberOfGradientMeasurements; ++i)
    {
      /** Show progress 0-100% */
      progressObserver->UpdateAndPrintProgress( i );

      /** Generate a perturbation, according to \mu_n ~ N( \mu_0, perturbationsigma^2 I )  */
      ParametersType perturbation = mu0;
      for (unsigned int p = 0; p < P; ++p)
      {
        perturbation[p] += perturbationSigma *
          randomgenerator->GetNormalVariate(0.0, 1.0);
      }

      /** Select new spatial samples for the computation of the metric */
      if ( stochasticgradients )
      {
        advmetric->SetImageSampler( randomsampler );
        this->SelectNewSamples();
      }

      /** Get approximate derivative */       
      try
      {
        this->GetScaledValueAndDerivative( perturbation, dummyvalue, approxgradient );      
      }
      catch( ExceptionObject& err )
      {
        this->m_StopCondition = MetricError;
        this->StopOptimization();
        throw err;
      }
      
      /* Compute magnitude. */
      approxgg += approxgradient.squared_magnitude();

      /** Get exact gradient and its magnitude */
      if ( stochasticgradients )
      {
        /** Set grid sampler */
        advmetric->SetImageSampler( gridsampler );

        /** Get derivative */
        try
        {
          this->GetScaledValueAndDerivative( perturbation, dummyvalue, exactgradient );      
        }
        catch( ExceptionObject& err )
        {
          this->m_StopCondition = MetricError;
          this->StopOptimization();
          throw err;
        }

        /** Compute error vector */
        diffgradient = exactgradient - approxgradient;

        /** Compute g^T g or g^T C^{-1}g, and e^T e or e^T C^{-1}e */
        if ( !maxlik )
        {
          /** g^T g and e^T e, if no maximum likelihood. */
          exactgg += exactgradient.squared_magnitude();
          diffgg += diffgradient.squared_magnitude();
        }
        else
        {
          /** compute g^T C^{-1} g */
          solveroutput = eig->D * ( exactgradient * eig->V );
          exactgg += solveroutput.squared_magnitude();
          solveroutput = eig->D * ( diffgradient * eig->V );
          diffgg += solveroutput.squared_magnitude();          
        }
      }
      else // no stochastic gradients
      {
        /** exact gradient equals approximate gradient */
        diffgg = 0.0;
        if ( !maxlik )
        {
          exactgg = approxgg;          
        }
        else
        {
          /** compute g^T C^{-1} g */
          solveroutput = eig->D * ( approxgradient * eig->V );
          exactgg += solveroutput.squared_magnitude();          
        } // end else: maxlik
      } // end else: no stochastic gradients
    } // end for

    progressObserver->PrintProgress( 1.0 );    

    /** Compute means */
    approxgg /= this->m_NumberOfGradientMeasurements;
    exactgg /= this->m_NumberOfGradientMeasurements;
    diffgg /= this->m_NumberOfGradientMeasurements;

    if (stochasticgradients)
    {
      /** Set back to what it was */
      advmetric->SetImageSampler( randomsampler );
    }    

    if ( randomCoordinateSampler.IsNotNull() )
    {
      /** Set back to what it was */
      randomCoordinateSampler->SetUseRandomSampleRegion( useRandomSampleRegion );
    }

    /** clean up */
    if (eig)
    {
      delete eig;
      eig = 0;
    }
    
    /** For output: gg and ee. 
     * gg and ee will be divided by Pd, but actually need to be divided by
     * the rank, in case of maximum likelihood. In case of no maximum likelihood,
     * the rank equals Pd. */
    gg = exactgg * Pd / static_cast<double>(rank);
    ee = diffgg * Pd / static_cast<double>(rank);
    
    return maxlik;

  } // end SampleGradients


  /** 
  * ******************** ComputeJacobianTerms **********************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
    ::ComputeJacobianTerms(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {
    std::string transformName = this->GetElastix()->
      GetElxTransformBase()->GetNameOfClass();

    const std::string translationName = "TranslationTransformElastix";
    const std::string bsplineName = "BSplineTransform";
    
    if ( transformName == translationName )
    {
      this->ComputeJacobianTermsTranslation(
        TrC, TrCC, maxJJ, maxJCJ );
    }
    else if ( transformName == bsplineName )
    {
      this->ComputeJacobianTermsBSpline(
        TrC, TrCC, maxJJ, maxJCJ );
    }
    else 
    {
      this->ComputeJacobianTermsGeneric(
        TrC, TrCC, maxJJ, maxJCJ );
    }   

  } // end ComputeJacobianTerms


  /** 
  * ************* ComputeJacobianTermsGeneric ****************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
    ::ComputeJacobianTermsGeneric(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {
    typedef typename CovarianceMatrixType::iterator     CovarianceMatrixIteratorType;
    typedef typename JacobianType::const_iterator       JacobianConstIteratorType;
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

    /** Initialize covariance matrix */
    this->m_CovarianceMatrix.SetSize( P,P );
    CovarianceMatrixType & cov = this->m_CovarianceMatrix;
    cov.Fill(0.0);    

    /** Loop over image and compute jacobian. Possibly apply scaling.
    * Compute C = 1/n \sum_i J_i^T J_i */
    std::vector<JacobianConstIteratorType> jacit(outdim);
    unsigned int samplenr = 0;
    CovarianceMatrixIteratorType covit;

    /** Prepare for progress printing */
    ProgressCommandPointer progressObserver = ProgressCommandType::New();
    progressObserver->SetUpdateFrequency( nrofsamples*2, 100 );
    progressObserver->SetStartString( "  Progress: " );
    elxout << "Computing JacobianTerms for " << this->elxGetClassName() 
      << " configuration... " << std::endl;

    for ( iter = begin; iter != end; ++iter )
    {
      /** Print progress 0-50% */
      progressObserver->UpdateAndPrintProgress( samplenr );
      ++samplenr;

      /** Read fixed coordinates and get jacobian. */      
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
    samplenr = 0;
    for ( iter = begin; iter != end; ++iter )
    {
      /** Show progress 50-100% */
      progressObserver->UpdateAndPrintProgress( samplenr + nrofsamples );
      ++samplenr;

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
          ParametersType jacjdy(jacj[dy], P, false);
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

    /** Finalize progress information */
    progressObserver->PrintProgress( 1.0 );

  } // end ComputeJacobianTermsGenericLinear


  /** 
  * ************* ComputeJacobianTermsTranslation ********************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
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
    void AdaptiveStochasticGradientDescent<TElastix>
    ::ComputeJacobianTermsBSpline(double & TrC, double & TrCC, 
    double & maxJJ, double & maxJCJ )
  {
    typedef typename JacobianType::const_iterator       JacobianConstIteratorType;
    typedef vnl_vector<double>                          JacobianColumnType;

    this->CheckForBSplineTransform();

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

    /** Initialize covariance matrix */
    this->m_CovarianceMatrix.SetSize( P,P );
    CovarianceMatrixType & cov = this->m_CovarianceMatrix;
    cov.Fill(0.0);    

    /** Loop over image and compute jacobian. 
    * Compute C = 1/n \sum_i J_i^T J_i 
    * Possibly apply scaling afterwards. */
    std::vector<JacobianConstIteratorType> jacit(outdim);
    unsigned int samplenr = 0;
    ParameterIndexArrayType & jacind = this->m_NonZeroJacobianIndices;
    const unsigned int sizejacind = jacind.GetSize();

    /** Prepare for progress printing */
    ProgressCommandPointer progressObserver = ProgressCommandType::New();
    progressObserver->SetUpdateFrequency( nrofsamples*2, 100 );
    progressObserver->SetStartString( "  Progress: " );
    elxout << "Computing JacobianTerms for " << this->elxGetClassName() 
      << " configuration... " << std::endl;

    for ( iter = begin; iter != end; ++iter )
    {
      /** Print progress 0-50% */
      progressObserver->UpdateAndPrintProgress( samplenr );
      ++samplenr;

      /** Read fixed coordinates and get jacobian.  */
      const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
      const JacobianType & jac = this->EvaluateBSplineTransformJacobian( point );     
      
      /** Update covariance matrix */
      for ( unsigned int pi = 0; pi < sizejacind; ++pi )
      {
        const unsigned int p = jacind[pi];

        const JacobianColumnType jaccolp = jac.get_column(pi);
        /** Initialize iterators at first column of (sparse) jacobian */
        for ( unsigned int d = 0; d < outdim; ++d)
        {
          jacit[d] = jac.begin() + d * sizejacind;
        }
        for ( unsigned int qi = 0; qi < sizejacind; ++qi )
        { 
          const unsigned int q = jacind[qi];
          for ( unsigned int d = 0; d < outdim; ++d)
          {
            cov[p][q] += jaccolp[d] * (*jacit[d]) / n;
            ++jacit[d];
          }                    
        } // qi
      } // pi       

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
    samplenr = 0;
    for ( iter = begin; iter != end; ++iter )
    {
      /** Show progress 50-100% */
      progressObserver->UpdateAndPrintProgress( samplenr + nrofsamples );
      ++samplenr;

      /** Read fixed coordinates and get jacobian.  */
      const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
      JacobianType jacj = this->EvaluateBSplineTransformJacobian( point );    

      /** Apply scales, if necessary */
      if ( this->GetUseScales() )
      {
        for (unsigned int pi = 0; pi < sizejacind; ++pi)
        {
          const unsigned int p = jacind[pi];
          jacj.scale_column( pi, 1.0/scales[p] );
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
          for (unsigned int pi = 0; pi < sizejacind; ++pi)
          {
            jacjjacj[dx][dy] += jacj[dx][pi] * jacj[dy][pi];
          } // pi
        } // dy
      } // dx
      JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

      /** Max_j [JJ] */
      maxJJ = vnl_math_max( maxJJ, JJ_j);

      /** Compute JCJ */
      double JCJ_j = 0.0;

      /** J_j C */
      JacobianType jacjC(outdim, sizejacind);
      jacjC.Fill(0.0);
      for( unsigned int dx = 0; dx < outdim; ++dx )
      {
        for ( unsigned int pi = 0; pi < sizejacind; ++pi)
        {
          const unsigned int p = jacind[pi];
          for ( unsigned int qi = 0; qi < sizejacind; ++qi)
          {
            const unsigned int q = jacind[qi];
            jacjC[dx][pi] += jacj[dx][qi] * cov[q][p];
          } // qi
        } // pi
      } // dx
  
      /** J_j C J_j^T */
      JacobianType jacjCjacj(outdim, outdim);
      for( unsigned int dx = 0; dx < outdim; ++dx )
      {
        ParametersType jacjCdx(jacjC[dx], sizejacind, false);
        for( unsigned int dy = 0; dy < outdim; ++dy )
        {
          ParametersType jacjdy(jacj[dy], sizejacind, false);
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

    /** Finalize progress information */
    progressObserver->PrintProgress( 1.0 );

  } // end ComputeJacobianTermsBSpline


  /** 
  * **************** SampleFixedImageForJacobianTerms *******************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
    ::SampleFixedImageForJacobianTerms(
    ImageSampleContainerPointer & sampleContainer )
  {
    /** Set up grid sampler */
    ImageSamplerPointer sampler = ImageSamplerType::New();
    sampler->SetInput( this->GetElastix()->
      GetElxMetricBase()->GetAsITKBaseType()->GetFixedImage() );
    sampler->SetInputImageRegion( this->GetElastix()->
      GetElxMetricBase()->GetAsITKBaseType()->GetFixedImageRegion() );
    sampler->SetMask( this->GetElastix()->
      GetElxMetricBase()->GetAsITKBaseType()->GetFixedImageMask() );

    /** Determine grid spacing of sampler such that the desired 
    * NumberOfJacobianMeasurements is achieved approximately.
    * Note that the actually obtained number of samples may be lower, due to masks.
    * This is taken into account at the end of this function. */
    unsigned int nrofsamples = this->m_NumberOfJacobianMeasurements;
    sampler->SetNumberOfSamples( nrofsamples );
    
    /** get samples and check the actually obtained number of samples */
    sampler->Update();
    sampleContainer = sampler->GetOutput();
    nrofsamples = sampleContainer->Size();
    if ( nrofsamples == 0 )
    {
      itkExceptionMacro(
        << "No valid voxels found to estimate the AdaptiveStochasticGradientDescent parameters." );
    }

  } // end SampleFixedImageForJacobianTerms


  /** 
  * **************** PrintSettingsVector **********************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
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


  /**
  * ****************** SaveCovarianceMatrix **********************
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
    ::SaveCovarianceMatrix( double sigma1, double sigma3, 
    const CovarianceMatrixType & cov )
  {
    if ( this->m_SaveCovarianceMatrix == false )
    {
      return;
    }

    /** Store covariance matrix in matlab format */
    unsigned int level = static_cast<unsigned int>(
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
    unsigned int elevel = this->GetConfiguration()->GetElastixLevel();

    /** Make filenames */
    std::ostringstream makeFileName("");
    makeFileName
      << this->GetConfiguration()->GetCommandLineArgument("-out")
      << "EstimatedCovarianceMatrix."
      << elevel
      << ".R" << level
      << ".mat";
    std::ostringstream makeCovName("");
    makeCovName
      << "EstCovE"
      << elevel
      << "R" << level;   
    std::ostringstream makeSigma1VarName("");
    makeSigma1VarName
      << "Sigma1E"
      << elevel
      << "R" << level;   
    std::ostringstream makeSigma3VarName("");
    makeSigma3VarName
      << "Sigma3E"
      << elevel
      << "R" << level;   

    /** Write to file */
    vnl_matlab_filewrite matlabWriter( makeFileName.str().c_str() );
    matlabWriter.write(cov, makeCovName.str().c_str() );
    matlabWriter.write(sigma1, makeSigma1VarName.str().c_str() );
    matlabWriter.write(sigma3, makeSigma3VarName.str().c_str() );

  } // end SaveCovarianceMatrix


  /**
  * ****************** CheckForBSplineTransform **********************
  * Check if the transform is of type BSplineDeformableTransform.
  * If so, we can speed up derivative calculations by only inspecting
  * the parameters in the support region of a point. 
  */

  template <class TElastix>
    void AdaptiveStochasticGradientDescent<TElastix>
    ::CheckForBSplineTransform( void )
  {
    this->m_TransformIsBSpline = false;
    typename TransformType::Pointer transform = this->GetRegistration()->
      GetAsITKBaseType()->GetTransform();
    this->m_NumberOfParameters = transform->GetNumberOfParameters();

    BSplineTransformType * testPtr1 = dynamic_cast<BSplineTransformType *>(
      transform.GetPointer() );
    if ( !testPtr1 )
    {
      this->m_BSplineTransform = 0;
      itkDebugMacro( "Transform is not BSplineDeformable" );
    }
    else
    {
      this->m_TransformIsBSpline = true;
      this->m_BSplineTransform = testPtr1;
      this->m_NumBSplineParametersPerDim = 
        this->m_BSplineTransform->GetNumberOfParametersPerDimension();
      this->m_NumBSplineWeights = this->m_BSplineTransform->GetNumberOfWeights();
      itkDebugMacro( "Transform is BSplineDeformable" );
    }

    /** Check if the transform is of type BSplineCombinationTransform. */
    this->m_TransformIsBSplineCombination = false;

    BSplineCombinationTransformType * testPtr2 = 
      dynamic_cast<BSplineCombinationTransformType *>( transform.GetPointer() );
    if ( !testPtr2 )
    {
      this->m_BSplineCombinationTransform = 0;
      itkDebugMacro( "Transform is not BSplineCombination" );
    }
    else
    {
      this->m_TransformIsBSplineCombination = true;
      this->m_BSplineCombinationTransform = testPtr2;

      /** The current transform in the BSplineCombinationTransform is 
      * always a BSplineTransform. */
      BSplineTransformType * bsplineTransform = 
        dynamic_cast<BSplineTransformType * >(
        this->m_BSplineCombinationTransform->GetCurrentTransform() );

      if ( !bsplineTransform )
      {
        itkExceptionMacro(<< "The BSplineCombinationTransform is not properly configured. The CurrentTransform is not set." );
      }
      this->m_NumBSplineParametersPerDim = 
        bsplineTransform->GetNumberOfParametersPerDimension();
      this->m_NumBSplineWeights = bsplineTransform->GetNumberOfWeights();
      itkDebugMacro( "Transform is BSplineCombination" );
    }

    /** Resize the weights and transform index arrays and compute the parameters offset. */
    if ( this->m_TransformIsBSpline || this->m_TransformIsBSplineCombination )
    {
      this->m_BSplineTransformWeights =
        BSplineTransformWeightsType( this->m_NumBSplineWeights );
      this->m_BSplineTransformIndices =
        BSplineTransformIndexArrayType( this->m_NumBSplineWeights );
      for ( unsigned int j = 0; j < FixedImageDimension; j++ )
      {
        this->m_BSplineParametersOffset[ j ] = j * this->m_NumBSplineParametersPerDim; 
      }
      this->m_NonZeroJacobianIndices.SetSize(
        FixedImageDimension * this->m_NumBSplineWeights );
      this->m_InternalTransformJacobian.SetSize( 
        FixedImageDimension, FixedImageDimension * this->m_NumBSplineWeights );
      this->m_InternalTransformJacobian.Fill( 0.0 );
    }
    else
    {   
      this->m_NonZeroJacobianIndices.SetSize( this->m_NumberOfParameters );
      for ( unsigned int i = 0; i < this->m_NumberOfParameters; ++i )
      {
        this->m_NonZeroJacobianIndices[ i ] = i;
      }
      this->m_InternalTransformJacobian.SetSize( 0, 0 );
    }

  } // end CheckForBSplineTransform
  

  /**
  * *************** EvaluateBSplineTransformJacobian ****************
  */

  template <class TElastix>
    const typename AdaptiveStochasticGradientDescent<TElastix>::TransformJacobianType &
    AdaptiveStochasticGradientDescent<TElastix>
    ::EvaluateBSplineTransformJacobian( 
    const FixedImagePointType & fixedImagePoint) const
  {
    typename MovingImageType::PointType dummy;
    bool sampleOk = false;
    if ( this->m_TransformIsBSpline )
    {
      this->m_BSplineTransform->TransformPoint( 
        fixedImagePoint,
        dummy,
        this->m_BSplineTransformWeights,
        this->m_BSplineTransformIndices,
        sampleOk );
    }
    else if ( this->m_TransformIsBSplineCombination )
    {
      this->m_BSplineCombinationTransform->TransformPoint( 
        fixedImagePoint,
        dummy,
        this->m_BSplineTransformWeights,
        this->m_BSplineTransformIndices,
        sampleOk );
    }
    if ( !sampleOk )
    {
      this->m_InternalTransformJacobian.Fill(0.0);
      this->m_NonZeroJacobianIndices.Fill(0);
      return this->m_InternalTransformJacobian;
    }

    /** If the transform is of type BSplineDeformableTransform or of type
    * BSplineCombinationTransform, we can obtain a speed up by only 
    * processing the affected parameters. */
    unsigned int i = 0;
    /** We assume the sizes of the m_InternalTransformJacobian and the
    * m_NonZeroJacobianIndices have already been set; Also we assume
    * that the InternalTransformJacobian is not 'touched' by other
    * functions (some elements always stay zero). */      
    for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
    {
      for ( unsigned int mu = 0; mu < this->m_NumBSplineWeights; mu++ )
      {
        /* The array weights contains the Jacobian values in a 1-D array 
        * (because for each parameter the Jacobian is non-zero in only 1 of the
        * possible dimensions) which is multiplied by the moving image gradient. */
        this->m_InternalTransformJacobian[ dim ][ i ] = this->m_BSplineTransformWeights[ mu ];

        /** The parameter number to which this partial derivative corresponds */
        const unsigned int parameterNumber = 
          this->m_BSplineTransformIndices[ mu ] + this->m_BSplineParametersOffset[ dim ];
        this->m_NonZeroJacobianIndices[ i ] = parameterNumber;

        /** Go to next column in m_InternalTransformJacobian */
        ++i;
      } //end mu for loop
    } //end dim for loop
  
    return this->m_InternalTransformJacobian;

  } // end EvaluateBSplineTransformJacobian

} // end namespace elastix

#endif // end #ifndef __elxAdaptiveStochasticGradientDescent_hxx

