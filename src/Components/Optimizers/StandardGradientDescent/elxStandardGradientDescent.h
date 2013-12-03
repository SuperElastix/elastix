/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxStandardGradientDescent_h
#define __elxStandardGradientDescent_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkStandardGradientDescentOptimizer.h"

namespace elastix
{


  /**
  * \class StandardGradientDescent
  * \brief A gradient descent optimizer with a decaying gain.
  *
  * This class is a wrap around the StandardGradientDescentOptimizer class.
  * It takes care of setting parameters and printing progress information.
  * For more information about the optimisation method, please read the documentation
  * of the StandardGradientDescentOptimizer class.
  *
  * The parameters used in this class are:
  * \parameter Optimizer: Select this optimizer as follows:\n
  *   <tt>(Optimizer "StandardGradientDescent")</tt>
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
  * \sa StandardGradientDescentOptimizer
  * \ingroup Optimizers
  */

  template <class TElastix>
  class StandardGradientDescent :
    public
    itk::StandardGradientDescentOptimizer,
    public
    OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef StandardGradientDescent             Self;
    typedef StandardGradientDescentOptimizer    Superclass1;
    typedef OptimizerBase<TElastix>             Superclass2;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( StandardGradientDescent, StandardGradientDescentOptimizer );

    /** Name of this class.
    * Use this name in the parameter file to select this specific optimizer.
    * example: <tt>(Optimizer "StandardGradientDescent")</tt>\n
    */
    elxClassNameMacro( "StandardGradientDescent" );

    /** Typedef's inherited from Superclass1, the StandardGradientDescentOptimizer.*/
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
    virtual void BeforeRegistration( void );
    virtual void BeforeEachResolution( void );
    virtual void AfterEachResolution( void );
    virtual void AfterEachIteration( void );
    virtual void AfterRegistration( void );

    /** Check if any scales are set, and set the UseScales flag on or off;
    * after that call the superclass' implementation */
    virtual void StartOptimization( void );

    /** Stop optimisation and pass on exception. */
    virtual void MetricErrorResponse( itk::ExceptionObject & err );

    /** Add SetCurrentPositionPublic, which calls the protected
    * SetCurrentPosition of the itkStandardGradientDescentOptimizer class.
    */
    virtual void SetCurrentPositionPublic( const ParametersType &param )
    {
      this->Superclass1::SetCurrentPosition( param );
    }

    /** Set the MaximumNumberOfSamplingAttempts. */
    itkSetMacro( MaximumNumberOfSamplingAttempts, unsigned long );

    /** Get the MaximumNumberOfSamplingAttempts. */
    itkGetConstReferenceMacro( MaximumNumberOfSamplingAttempts, unsigned long );

  protected:

    StandardGradientDescent();
    virtual ~StandardGradientDescent() {};


  private:

    StandardGradientDescent( const Self& ); // purposely not implemented
    void operator=( const Self& );          // purposely not implemented

    /** Private variables for the sampling attempts. */
    unsigned long m_MaximumNumberOfSamplingAttempts;
    unsigned long m_CurrentNumberOfSamplingAttempts;
    unsigned long m_PreviousErrorAtIteration;

  };


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxStandardGradientDescent.hxx"
#endif

#endif // end #ifndef __elxStandardGradientDescent_h
