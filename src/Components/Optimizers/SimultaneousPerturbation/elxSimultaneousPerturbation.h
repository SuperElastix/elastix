/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxSimultaneousPerturbation_h
#define __elxSimultaneousPerturbation_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkSPSAOptimizer.h"

namespace elastix
{

  /**
   * \class SimultaneousPerturbation
   * \brief An optimizer based on the itk::SPSAOptimizer.
   *
   * The ITK doxygen help gives more information about this optimizer.
   *
   * This optimizer supports the NewSamplesEveryIteration parameter.
   *
   * The parameters used in this class are:
   * \parameter Optimizer: Select this optimizer as follows:\n
   *    <tt>(Optimizer "SimultaneousPerturbation")</tt>
   * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
   *    example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
   *    Default value: 500.
   * \parameter NumberOfPerturbations: The number of perturbation used to
   *    construct a gradient estimate \f$g_k\f$. \n
   *    \f$q =\f$ NumberOfPerturbations \n
   *    \f$g_k = 1/q \sum_{j = 1..q} g^(j)_k\f$ \n
   *    This parameter can be defined for each resolution. \n
   *    example: <tt>(NumberOfPerturbations 1 1 2)</tt> \n
   *    Default value: 1.
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
   * \parameter SP_c: The perturbation step size \f$c(k)\f$ at each iteration \f$k\f$ is defined by \n
   *   \f$c(k) =  SP\_c / ( k + 1)^{SP\_gamma}\f$. \n
   *   SP_c can be defined for each resolution. \n
   *   example: <tt>(SP_c 2.0 1.0 1.0)</tt> \n
   *   The default value is 1.0.
   * \parameter SP_gamma: The perturbation step size \f$c(k)\f$ at each iteration \f$k\f$ is defined by \n
   *   \f$c(k) =  SP\_c / ( k + 1)^{SP\_gamma}\f$. \n
   *   SP_gamma can be defined for each resolution. \n
   *   example: <tt>(SP_gamma 0.101 0.101 0.101)</tt> \n
   *   The default/recommended value is 0.101.
   * \parameter ShowMetricValues: Defines whether to compute/show the metric value in each iteration. \n
   *   This flag can NOT be defined for each resolution. \n
   *   example: <tt>(ShowMetricValues "true" )</tt> \n
   *   Default value: "false". Note that turning this flag on increases computation time.
   *
   *
   * \ingroup Optimizers
   */

  template <class TElastix>
    class SimultaneousPerturbation :
    public
      itk::SPSAOptimizer,
    public
      OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef SimultaneousPerturbation            Self;
    typedef SPSAOptimizer                       Superclass1;
    typedef OptimizerBase<TElastix>             Superclass2;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( SimultaneousPerturbation, SPSAOptimizer );

    /** Name of this class.
     * Use this name in the parameter file to select this specific optimizer. \n
     * example: <tt>(Optimizer "SimultaneousPerturbation")</tt>\n
     */
    elxClassNameMacro( "SimultaneousPerturbation" );

    /** Typedef's inherited from Superclass1.*/
    typedef Superclass1::CostFunctionType     CostFunctionType;
    typedef Superclass1::CostFunctionPointer  CostFunctionPointer;
    typedef Superclass1::StopConditionType    StopConditionType;

    /** Typedef's inherited from Elastix.*/
    typedef typename Superclass2::ElastixType           ElastixType;
    typedef typename Superclass2::ElastixPointer        ElastixPointer;
    typedef typename Superclass2::ConfigurationType     ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer  ConfigurationPointer;
    typedef typename Superclass2::RegistrationType      RegistrationType;
    typedef typename Superclass2::RegistrationPointer   RegistrationPointer;
    typedef typename Superclass2::ITKBaseType           ITKBaseType;

    /** Typedef for the ParametersType. */
    typedef typename Superclass1::ParametersType        ParametersType;

    /** Methods that take care of setting parameters and printing progress information.*/
    virtual void BeforeRegistration(void);
    virtual void BeforeEachResolution(void);
    virtual void AfterEachResolution(void);
    virtual void AfterEachIteration(void);
    virtual void AfterRegistration(void);

    /** Override the SetInitialPosition.
     * Override the implementation in itkOptimizer.h, to
     * ensure that the scales array and the parameters
     * array have the same size. */
    virtual void SetInitialPosition( const ParametersType & param );

  protected:

      SimultaneousPerturbation();
      virtual ~SimultaneousPerturbation() {};

      bool m_ShowMetricValues;

  private:

      SimultaneousPerturbation( const Self& );  // purposely not implemented
      void operator=( const Self& );              // purposely not implemented

  }; // end class SimultaneousPerturbation


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxSimultaneousPerturbation.hxx"
#endif

#endif // end #ifndef __elxSimultaneousPerturbation_h
