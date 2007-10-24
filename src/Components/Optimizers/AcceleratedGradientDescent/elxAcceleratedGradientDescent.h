#ifndef __elxAcceleratedGradientDescent_h
#define __elxAcceleratedGradientDescent_h

#include "itkAcceleratedGradientDescentOptimizer.h"
#include "itkImageGridSampler.h"
#include "elxIncludes.h"
#include "elxProgressCommand.h"

namespace elastix
{
  using namespace itk;


  /**
  * \class AcceleratedGradientDescent
  * \brief A gradient descent optimizer with an adaptive gain.
  *
  * This class is a wrap around the AcceleratedGradientDescentOptimizer class.
  * It takes care of setting parameters and printing progress information.
  * For more information about the optimisation method, please read the documentation
  * of the AcceleratedGradientDescentOptimizer class.
  *
  * The parameters used in this class are:
  * \parameter Optimizer: Select this optimizer as follows:\n
  *   <tt>(Optimizer "AcceleratedGradientDescent")</tt>
  * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
  *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
  *    Default value: 100.
  * \parameter SP_a: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
  *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
  *   SP_a can be defined for each resolution. \n
  *   example: <tt>(SP_a 3200.0 3200.0 1600.0)</tt> \n
  *   The default value is 400.0. Tuning this variable for you specific problem is recommended.
  * \parameter SP_A: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
  *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
  *   SP_A can be defined for each resolution. \n
  *   example: <tt>(SP_A 50.0 50.0 100.0)</tt> \n
  *   The default/recommended value is 50.0.
  * \parameter SP_alpha: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
  *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
  *   SP_alpha can be defined for each resolution. \n
  *   example: <tt>(SP_alpha 0.602 0.602 0.602)</tt> \n
  *   The default/recommended value is 0.602.
  *
  * \todo: document extra parameters
  *
  * \sa AcceleratedGradientDescentOptimizer
  * \ingroup Optimizers
  */

  template <class TElastix>
  class AcceleratedGradientDescent :
    public
    itk::AcceleratedGradientDescentOptimizer,
    public
    OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef AcceleratedGradientDescent          Self;
    typedef AcceleratedGradientDescentOptimizer Superclass1;
    typedef OptimizerBase<TElastix>             Superclass2;
    typedef SmartPointer<Self>                  Pointer;
    typedef SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( AcceleratedGradientDescent, AcceleratedGradientDescentOptimizer );

    /** Name of this class.
    * Use this name in the parameter file to select this specific optimizer.
    * example: <tt>(Optimizer "AcceleratedGradientDescent")</tt>\n
    */
    elxClassNameMacro( "AcceleratedGradientDescent" );

    /** Typedef's inherited from Superclass1, the AcceleratedGradientDescentOptimizer.*/
    typedef Superclass1::CostFunctionType     CostFunctionType;
    typedef Superclass1::CostFunctionPointer  CostFunctionPointer;
    typedef Superclass1::StopConditionType    StopConditionType;

    /** Typedef's inherited from Superclass2, the elastix OptimizerBase .*/
    typedef typename Superclass2::ElastixType           ElastixType;
    typedef typename Superclass2::ElastixPointer        ElastixPointer;
    typedef typename Superclass2::ConfigurationType     ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer  ConfigurationPointer;
    typedef typename Superclass2::RegistrationType      RegistrationType;
    typedef typename Superclass2::RegistrationPointer   RegistrationPointer;
    typedef typename Superclass2::ITKBaseType           ITKBaseType;

    /** Typedef for the ParametersType. */
    typedef typename Superclass1::ParametersType        ParametersType;

    /** Methods invoked by elastix, in which parameters can be set and 
    * progress information can be printed. */
    virtual void BeforeRegistration(void);
    virtual void BeforeEachResolution(void);
    virtual void AfterEachResolution(void);
    virtual void AfterEachIteration(void);
    virtual void AfterRegistration(void);   

    /** Check if any scales are set, and set the UseScales flag on or off; 
    * after that call the superclass' implementation */
    virtual void StartOptimization(void);

    /** If automatic gain estimation is desired, then estimate SP_a, SP_alpha
     * SigmoidScale, SigmoidMax, SigmoidMin.
     * After that call Superclass' implementation.  */
    virtual void ResumeOptimization(void);

    /** Call the superclass' implementation and check if the minimum step length
     * stopping condition is satisfied. */
    virtual void AdvanceOneStep(void);

    /** Set/Get whether automatic parameter estimation is desired. 
     * If true, make sure to set the maximum step length.
     *
     * The following parameters are automatically determined:
     * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1),
     * SigmoidScale. 
     * A usually suitable value for SP_A is 25. This has
     * to be set manually though.
     * \todo: AutomaticParameterEstimation does not work in combination
     * with the MultiMetricMultiResolutionRegistration component.
     */
    itkSetMacro(AutomaticParameterEstimation, bool);
    itkGetConstMacro(AutomaticParameterEstimation, bool);

    /** Set/Get maximum step length */
    itkSetMacro( MaximumStepLength, double );
    itkGetConstMacro( MaximumStepLength, double );

    /** Set/Get minimum step length; this is used as a stopping criterion.
     * When the gain falls below minsteplength/maxsteplength * gain(0)
     * the optimisation is stopped. Set it to zero if you do not want to
     * use this stopping criterion. */
    itkSetMacro( MinimumStepLength, double );
    itkGetConstMacro( MinimumStepLength, double );

  protected:

    typedef typename RegistrationType::FixedImageType   FixedImageType;
    typedef typename RegistrationType::MovingImageType  MovingImageType;
    typedef typename FixedImageType::RegionType         FixedImageRegionType;
    typedef typename FixedImageType::IndexType          FixedImageIndexType;
    typedef typename FixedImageType::PointType          FixedImagePointType;
    typedef typename RegistrationType::ITKBaseType      itkRegistrationType;
    typedef typename itkRegistrationType::TransformType TransformType;
    typedef typename TransformType::JacobianType        JacobianType;
    typedef typename JacobianType::ValueType            JacobianValueType;
    struct SettingsType { double a, A, alpha, fmax, fmin, omega; };
    typedef typename std::vector<SettingsType>          SettingsVectorType;
    
    typedef ImageGridSampler< FixedImageType >          ImageSamplerType;
    typedef typename ImageSamplerType::Pointer          ImageSamplerPointer;
    typedef typename 
      ImageSamplerType::ImageSampleContainerType        ImageSampleContainerType;
    typedef typename ImageSampleContainerType::Pointer  ImageSampleContainerPointer;
    typedef ProgressCommand                             ProgressCommandType;
    typedef typename ProgressCommand::Pointer           ProgressCommandPointer;
    typedef Array2D<double>                             CovarianceMatrixType; 
    
    AcceleratedGradientDescent();
    virtual ~AcceleratedGradientDescent() {};
    
    /** Variable to store the automatically determined settings for each resolution */
    SettingsVectorType m_SettingsVector;

    /** Some options for automatic parameter estimation */
    unsigned int m_NumberOfGradientMeasurements;
    unsigned int m_NumberOfJacobianMeasurements;
    unsigned int m_NumberOfSamplesForExactGradient;
    std::string  m_JacobianTermComputationMethod;
    CovarianceMatrixType m_CovarianceMatrix;

    /** Print the contents of the settings vector to elxout */
    virtual void PrintSettingsVector( const SettingsVectorType & settings ) const;

    /** Estimates some reasonable values for the parameters
     * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and
     * SigmoidScale. */
    virtual void AutomaticParameterEstimation( void );

    /** Measure some derivatives, exact and approximated. Returns
     * the squared magnitude of the gradient and approximation error.
     * Needed for the automatic parameter estimation.
     * Gradients are measured at position mu_n, which are generated according to:
     * mu_n - mu_0 ~ N(0, perturbationSigma^2 I );
     */
    virtual void SampleGradients(const ParametersType & mu0,
      double perturbationSigma, double & gg, double & ee);

    /** Returns a comtainer of fixed image samples, sampled using a grid sampler
     * The grid size is determined from the user entered number of jacobian measurements,
     * or a default value of 200 is used.
     * The actual number of samples depends on the presence of masks, and 
     * the restriction that the gridspacing of the gridsampler must be integer.
     * The samples input variable contains the sample container after execution.
     * It does not have to be initialised/allocated before.
     */
    virtual void SampleFixedImageForJacobianTerms(
      ImageSampleContainerPointer & sampleContainer );


    /** Functions to compute the jacobian terms needed for the automatic
     * parameter estimation */
    virtual void ComputeJacobianTerms(double & TrC, double & TrCC, 
      double & maxJJ, double & maxJCJ );

    /** Implementation of the jacobian terms, using a method that is
     * quadratically complex regarding the number of jacobian measurements.
     * The memory usage is linear proportional to the number of jacobian measurements
     * and linear proportional to the number of parameters. */
    virtual void ComputeJacobianTermsGenericQuadratic(double & TrC, double & TrCC, 
      double & maxJJ, double & maxJCJ );

    /** Implementation of the jacobian terms, using a method that is
     * linearly complex regarding the number of jacobian measurements.
     * The memory usage is independent on the number of jacobian measurements
     * and quadratically proportional to the number of parameters. */
    virtual void ComputeJacobianTermsGenericLinear(double & TrC, double & TrCC, 
      double & maxJJ, double & maxJCJ );
    
    virtual void ComputeJacobianTermsAffine(double & TrC, double & TrCC, 
      double & maxJJ, double & maxJCJ );

    virtual void ComputeJacobianTermsTranslation(double & TrC, double & TrCC, 
      double & maxJJ, double & maxJCJ );

    virtual void ComputeJacobianTermsBSpline(double & TrC, double & TrCC, 
      double & maxJJ, double & maxJCJ );
    
  private:

    AcceleratedGradientDescent( const Self& );  // purposely not implemented
    void operator=( const Self& );              // purposely not implemented

    bool m_AutomaticParameterEstimation;
    double m_MaximumStepLength;      
    double m_MinimumStepLength;

  }; // end class AcceleratedGradientDescent


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAcceleratedGradientDescent.hxx"
#endif

#endif // end #ifndef __elxAcceleratedGradientDescent_h
