/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxPreconditionedGradientDescent_h
#define __elxPreconditionedGradientDescent_h

#include "itkStochasticPreconditionedGradientDescentOptimizer.h"
#include "elxIncludes.h"

namespace elastix
{
  using namespace itk;


  /**
  * \class PreconditionedGradientDescent
  * \brief A gradient descent optimizer with a decaying gain.
  *
  * This class is a wrap around the StochasticPreconditionedGradientOptimizer class.
  * It takes care of setting parameters and printing progress information.
  * For more information about the optimisation method, please read the documentation
  * of the StochasticPreconditionedGradientOptimizer class.
  *
  * The parameters used in this class are:
  * \parameter Optimizer: Select this optimizer as follows:\n
  *   <tt>(Optimizer "PreconditionedGradientDescent")</tt>
  * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
  *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
  *    Default/recommended value: 500.
  * \parameter MaximumNumberOfSamplingAttempts: The maximum number of sampling attempts. Sometimes
  *   not enough corresponding samples can be drawn, upon which an exception is thrown. With this
  *   parameter it is possible to try to draw another set of samples. \n
  *   example: <tt>(MaximumNumberOfSamplingAttempts 10 15 10)</tt> \n
  *    Default value: 0, i.e. just fail immediately, for backward compatibility.
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
  * \sa StochasticPreconditionedGradientOptimizer
  * \ingroup Optimizers
  */

  template <class TElastix>
  class PreconditionedGradientDescent :
    public
    itk::StochasticPreconditionedGradientDescentOptimizer,
    public
    OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef PreconditionedGradientDescent             Self;
    typedef StochasticPreconditionedGradientDescentOptimizer    Superclass1;
    typedef OptimizerBase<TElastix>             Superclass2;
    typedef SmartPointer<Self>                  Pointer;
    typedef SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( PreconditionedGradientDescent, StochasticPreconditionedGradientDescentOptimizer );

    /** Name of this class.
    * Use this name in the parameter file to select this specific optimizer.
    * example: <tt>(Optimizer "PreconditionedGradientDescent")</tt>\n
    */
    elxClassNameMacro( "PreconditionedGradientDescent" );

    /** Typedef's inherited from Superclass1, the StochasticPreconditionedGradientDescentOptimizer.*/
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

    /** Some typedefs for computing the SelfHessian */
    typedef typename Superclass::PreconditionValueType     PreconditionValueType;
    typedef typename Superclass::PreconditionType          PreconditionType;
    typedef typename Superclass::EigenSystemType           EigenSystemType;

    /** Methods invoked by elastix, in which parameters can be set and 
    * progress information can be printed. */
    virtual void BeforeRegistration( void );
    virtual void BeforeEachResolution( void );
    virtual void AfterEachResolution( void );
    virtual void AfterEachIteration( void );
    virtual void AfterRegistration( void );   

    /** Check if any scales are set, and set the UseScales flag on or off; 
    * after that call the superclass' implementation */
    virtual void StartOptimization( void );

    /** Stop optimisation and pass on exception. */
    virtual void MetricErrorResponse( ExceptionObject & err );

    /** Add SetCurrentPositionPublic, which calls the protected
    * SetCurrentPosition of the itkStochasticPreconditionedGradientDescentOptimizer class.
    */
    virtual void SetCurrentPositionPublic( const ParametersType &param )
    {
      this->Superclass1::SetCurrentPosition( param );
    }

    /** Set the MaximumNumberOfSamplingAttempts. */
    itkSetMacro( MaximumNumberOfSamplingAttempts, unsigned long );

    /** Get the MaximumNumberOfSamplingAttempts. */
    itkGetConstReferenceMacro( MaximumNumberOfSamplingAttempts, unsigned long );

    /** Set the SelfHessian as a preconditioning matrix and call Superclass' implementation.
     * Only done when m_PreconditionMatrixSet == false; */
    virtual void ResumeOptimization( void );

  protected:

    PreconditionedGradientDescent();
    virtual ~PreconditionedGradientDescent() {};

    /** Get the SelfHessian from the MeanSquares metric and submit as Precondition matrix */
    virtual void SetSelfHessian( void );

  private:

    PreconditionedGradientDescent( const Self& ); // purposely not implemented
    void operator=( const Self& );          // purposely not implemented

    /** Private variables for the sampling attempts. */
    unsigned long m_MaximumNumberOfSamplingAttempts;
    unsigned long m_CurrentNumberOfSamplingAttempts;
    unsigned long m_PreviousErrorAtIteration;

    /** Private variables for self hessian support. */
    bool          m_PreconditionMatrixSet;
    unsigned int  m_NumberOfSamplesForSelfHessian;
    double        m_SelfHessianSmoothingSigma;

  }; // end class PreconditionedGradientDescent


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxPreconditionedGradientDescent.hxx"
#endif

#endif // end #ifndef __elxPreconditionedGradientDescent_h
