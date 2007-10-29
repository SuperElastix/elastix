#ifndef __elxAcceleratedGradientDescent_hxx
#define __elxAcceleratedGradientDescent_hxx

#include "elxAcceleratedGradientDescent.h"
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include "vnl/vnl_math.h"
#include "vnl/algo/vnl_cholesky.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_matlab_filewrite.h"
#include "itkImageRandomConstIteratorWithIndex.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkAdvancedImageToImageMetric.h"




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
    this->m_MinimumStepLength = 0.0;

    this->m_NumberOfGradientMeasurements = 0;
    this->m_NumberOfJacobianMeasurements = 0;
    this->m_NumberOfSamplesForExactGradient = 100000;

    this->m_JacobianTermComputationMethod = "Linear";
    this->m_UseMaximumLikelihoodMethod = false;

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

      /** Read minimum step length. Default = 0.1 * maxsteplength. */
      this->m_MinimumStepLength = 0.1 * this->m_MaximumStepLength;
      this->GetConfiguration()->ReadParameter( this->m_MinimumStepLength,
        "MinimumStepLength", this->GetComponentLabel(), level, 0 );

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

      this->m_JacobianTermComputationMethod = "Linear";
      this->GetConfiguration()->ReadParameter( 
        this->m_JacobianTermComputationMethod,
        "JacobianTermComputationMethod",
        this->GetComponentLabel(), level, 0 );

      if ( this->m_JacobianTermComputationMethod == "Linear" )
      {
        this->m_UseMaximumLikelihoodMethod = false;
        this->GetConfiguration()->ReadParameter( 
          this->m_UseMaximumLikelihoodMethod,
          "UseMaximumLikelihoodMethod",
          this->GetComponentLabel(), level, 0 );
      }      

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
  * ********************** AdvanceOneStep **********************
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::AdvanceOneStep(void)
  {
    /** Call the superclass' implementation */
    this->Superclass1::AdvanceOneStep();

    const double minGainFraction = 
      this->GetMinimumStepLength() / this->GetMaximumStepLength();

    const double gain0 = this->Compute_a( 0.0 );
    const double gainNextIt = this->Compute_a( this->GetCurrentTime() );

    /** Stop the optimization when the gain is too small */
    if ( gainNextIt/gain0 < minGainFraction )
    {
      this->m_StopCondition = MinimumStepSize;
      this->StopOptimization();
    }

  } // end AdvanceOneStep


  /** 
  * ******************* AutomaticParameterEstimation **********************
  * Estimates some reasonable values for the parameters
  * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and SigmoidScale. 
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
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
    const double sigma4 = delta / vcl_sqrt( maxJJ );
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

    /** Store covariance matrix in matlab format */
    unsigned int level = static_cast<unsigned int>(
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

    /** Make filenames */
    std::ostringstream makeFileName("");
    makeFileName
      << this->GetConfiguration()->GetCommandLineArgument("-out")
      << "EstimatedCovarianceMatrix."
      << this->GetConfiguration()->GetElastixLevel()
      << ".R" << level
      << ".mat";
    std::ostringstream makeCovName("");
    makeCovName
      << "EstCovE"
      << this->GetConfiguration()->GetElastixLevel()
      << "R" << level;   
    std::ostringstream makeSigma1VarName("");
    makeSigma1VarName
      << "Sigma1E"
      << this->GetConfiguration()->GetElastixLevel()
      << "R" << level;   
    std::ostringstream makeSigma3VarName("");
    makeSigma3VarName
      << "Sigma3E"
      << this->GetConfiguration()->GetElastixLevel()
      << "R" << level;   

    /** Write to file */
    vnl_matlab_filewrite matlabWriter( makeFileName.str().c_str() );
    matlabWriter.write(this->m_CovarianceMatrix, makeCovName.str().c_str() );
    matlabWriter.write(sigma1, makeSigma1VarName.str().c_str() );
    matlabWriter.write(sigma3, makeSigma3VarName.str().c_str() );

    /** Clean up */
    this->m_CovarianceMatrix.SetSize(0,0);

    const double alpha = 1.0;
    const double A = this->GetParam_A();
    const double a_max = A * delta / sigma1  / vcl_sqrt( maxJCJ );
    const double noisefactor = sigma1*sigma1 / ( sigma1*sigma1 + sigma3*sigma3 + 1e-14 );
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
    bool AcceleratedGradientDescent<TElastix>
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
    typedef vnl_svd<double>                             SVDType;

    /** Some shortcuts */
    const unsigned int P = static_cast<unsigned int>( mu0.GetSize() );
    const double Pd = static_cast<double>( P );
    CovarianceMatrixType & cov = this->m_CovarianceMatrix;

    /** Prepare for maximum likelihood estimation of sigmas. In that case we 
    * need a cholesky matrix decomposition */
    vnl_cholesky * cholesky = 0;
    SVDType * svd = 0;
    bool maxlik = false;
    bool useSVD = false;
    if ( (cov.size() != 0) && this->m_UseMaximumLikelihoodMethod )
    {
      maxlik = true;
      cholesky = new vnl_cholesky(cov, vnl_cholesky::estimate_condition);
      if ( cholesky->rcond() < 1e-6 // sqrt(machineprecision)
        || cholesky->rcond() > 1.1  // happens when some eigenvalues are 0 or -0
        || cholesky->rank_deficiency() ) // if !=0 something is wrong
      {
        xl::xout["warning"] << "WARNING: Covariance matrix is singular! Using SVD instead of Cholesky." << std::endl;
        delete cholesky;
        cholesky = 0;   
        useSVD = true;
        svd = new SVDType( cov, -1e-6 );
      }      
    }

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
    ImageRandomSamplerPointer randomsampler = 0;
    AdvancedMetricPointer advmetric = 0;
    unsigned int normalnumberofsamples = 0;

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
      advmetric = dynamic_cast<AdvancedMetricType * >( 
        this->GetElastix()->GetElxMetricBase() );
      if (advmetric)
      {
        randomsampler = dynamic_cast<ImageRandomSamplerType*>( advmetric->GetImageSampler() );
        if ( (!advmetric->GetUseImageSampler()) || randomsampler.IsNull() )
        {
          stochasticgradients = false;
        }
      }
      else
      {
        stochasticgradients = false;
      }
    }     

    /** Set up the grid samper for the "exact" gradients */
    ImageSamplerPointer gridsampler = 0;

    if (stochasticgradients)
    {
      gridsampler = ImageSamplerType::New();
      gridsampler->SetInput( randomsampler->GetInput() );
      gridsampler->SetInputImageRegion( randomsampler->GetInputImageRegion() );
      gridsampler->SetMask( randomsampler->GetMask() );
      /** Compute the grid spacing */
      unsigned int allsamples = this->m_NumberOfSamplesForExactGradient;
      const double fixdimd = static_cast<double>( 
        randomsampler->GetInputImageRegion().GetImageDimension() );
      const double fraction = 
        static_cast<double>( randomsampler->GetInputImageRegion().GetNumberOfPixels() ) /
        static_cast<double>( allsamples );
      int gridspacing = static_cast<int>( 
        vnl_math_rnd( vcl_pow(fraction, 1.0/fixdimd) )   );
      gridspacing = vnl_math_max( 1, gridspacing );
      typename ImageSamplerType::SampleGridSpacingType gridspacings;
      gridspacings.Fill( gridspacing );
      gridsampler->SetSampleGridSpacing( gridspacings );
      gridsampler->Update();
    }

    /** Prepare for progress printing */
    ProgressCommandPointer progressObserver = ProgressCommandType::New();
    progressObserver->SetUpdateFrequency( numberofgradients, numberofgradients );
    progressObserver->SetStartString( "  Progress: " );
    elxout << "Sampling gradients for " << this->elxGetClassName() 
      << " configuration... " << std::endl;

    /** Compute gg for some random parameters */      
    typename RandomGeneratorType::Pointer randomgenerator = RandomGeneratorType::New();
    for ( unsigned int i = 0 ; i < numberofgradients; ++i)
    {
      /** Show progress 0-100% */
      progressObserver->UpdateAndPrintProgress( i );

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
        this->SelectNewSamples();
        advmetric->SetImageSampler( randomsampler );
      }

      /** Get approximate derivative and its magnitude */
      this->GetScaledValueAndDerivative( perturbation, dummyvalue, approxgradient );
      approxgg += approxgradient.squared_magnitude();

      /** Get exact gradient and its magnitude */
      if ( stochasticgradients )
      {
        advmetric->SetImageSampler( gridsampler );
        this->GetScaledValueAndDerivative( perturbation, dummyvalue, exactgradient );
        diffgradient = exactgradient - approxgradient;

        if ( !maxlik )
        {
          exactgg += exactgradient.squared_magnitude();
          diffgg += diffgradient.squared_magnitude();
        }
        else
        {
          /** compute g^T C^{-1} g */
          if ( useSVD )
          {            
            solveroutput = svd->solve( exactgradient );
            exactgg += dot_product( exactgradient, solveroutput);
            solveroutput = svd->solve( diffgradient );
            diffgg += dot_product( diffgradient, solveroutput);
          }
          else
          {
            cholesky->solve( exactgradient, &solveroutput );
            exactgg += dot_product( exactgradient, solveroutput);
            cholesky->solve( diffgradient, &solveroutput );
            diffgg += dot_product( diffgradient, solveroutput);
          }
        }
      }
      else
      {
        exactgg = approxgg;
        diffgg = 0.0;
      }
    } // end for

    progressObserver->PrintProgress( 1.0 );    

    approxgg /= numberofgradients;
    exactgg /= numberofgradients;
    diffgg /= numberofgradients;

    if (stochasticgradients)
    {
      /** Set back to what it was */
      advmetric->SetImageSampler( randomsampler );
    }    

    /** For output: */
    gg = exactgg;
    ee = diffgg;

    /** clean up */
    if (cholesky)
    {
      delete cholesky;
      cholesky = 0;
    }
    if (svd)
    {
      delete svd;
      svd = 0;
    }

    return maxlik;

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
    const std::string bsplineName = "BSplineTransform";
    const std::string linearMethod = "Linear";
    const std::string quadraticMethod = "Quadratic";

    if ( transformName == translationName )
    {
      this->ComputeJacobianTermsTranslation(
        TrC, TrCC, maxJJ, maxJCJ );
    }
    else if ( (transformName == bsplineName) &&
      (this->m_JacobianTermComputationMethod == linearMethod) )
    {
      this->ComputeJacobianTermsBSpline(
        TrC, TrCC, maxJJ, maxJCJ );
    }
    else if ( this->m_JacobianTermComputationMethod == linearMethod ) 
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

    /** Prepare for progress printing */
    ProgressCommandPointer progressObserver = ProgressCommandType::New();
    progressObserver->SetUpdateFrequency( nrofsamples, 100 );
    progressObserver->SetStartString( "  Progress: " );
    elxout << "Sampling Jacobians for " << this->elxGetClassName() 
      << " configuration... " << std::endl;

    /** Loop over image and compute jacobian. Save the jacobians in a vector. */
    unsigned int s = 0;
    for ( iter = begin; iter != end; ++iter )
    {
      progressObserver->UpdateAndPrintProgress( s );

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
    progressObserver->PrintProgress( 1.0 );

    elxout << "Computing JacobianTerms for " << this->elxGetClassName() 
      << " configuration... " << std::endl;

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
      progressObserver->UpdateAndPrintProgress( j );

      /** Get jacobian */
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
        ParametersType jacjdx(jacj[dx], P, false);
        for( unsigned int dy = 0; dy < outdim; ++dy )
        {
          for (unsigned int p = 0; p < P; ++p)
          {
            ParametersType jacjdy(jacj[dy], P, false);
            jacjjacj(dx,dy)= dot_product(jacjdx, jacjdy);
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
            ParametersType jacidy(jaci[dy], P, false);
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

    progressObserver->PrintProgress( 1.0 );

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


  /**
  * ****************** CheckForBSplineTransform **********************
  * Check if the transform is of type BSplineDeformableTransform.
  * If so, we can speed up derivative calculations by only inspecting
  * the parameters in the support region of a point. 
  */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
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
    const typename AcceleratedGradientDescent<TElastix>::TransformJacobianType &
    AcceleratedGradientDescent<TElastix>
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

#endif // end #ifndef __elxAcceleratedGradientDescent_hxx

