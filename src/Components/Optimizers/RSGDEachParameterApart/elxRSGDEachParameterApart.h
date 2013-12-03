/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxRSGDEachParameterApart_h
#define __elxRSGDEachParameterApart_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkRSGDEachParameterApartOptimizer.h"

namespace elastix
{

  /**
   * \class RSGDEachParameterApart
   * \brief An optimizer based on gradient descent.
   *
   * The underlying itk class is almost a copy of the normal
   * RegularStepGradientDescent. The difference is that each
   * parameter has its own step length, whereas the normal
   * RSGD has one step length that is used for all parameters.
   *
   * This could cause inaccuracies, if, for example, parameter
   * 1, 2 and 3 are already close to the optimum, but parameter
   * 4 not yet. The average stepsize is halved then, so parameter
   * 4 will not have time to reach its optimum (in a worst case
   * scenario).
   *
   * The RSGDEachParameterApart stops only if ALL steplenghts
   * are smaller than the MinimumStepSize given in the parameter
   * file!
   *
   * The elastix shell class (so, this class...), is a copy of
   * the elxRegularStepGradientDescent, so the parameters in the
   * parameter file, the output etc are similar.
   *
   * The parameters used in this class are:
   * \parameter Optimizer: Select this optimizer as follows:\n
   *    <tt>(Optimizer "RSGDEachParameterApart")</tt>
   * \parameter MaximumNumberOfIterations: the maximum number of iterations in each resolution. \n
   *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
   *   Default value: 100.
   * \parameter MinimumGradientMagnitude: stopping criterion. If the magnitude of the derivative
   *   of the cost function is below this value, optimisation is stopped. \n
   *   example: <tt>(MinimumGradientMagnitude 0.0001 0.0001 0.001)</tt> \n
   *   Default value: 1e-8.
   * \parameter MinimumStepLength: stopping criterion. If the steplength is below this
   *   value, optimisation is stopped. \n
   *   example: <tt>(MinimumStepLength 1.0 0.5 0.1)</tt> \n
   *   Default value: <em>0.5 / 2^resolutionlevel</em>
   * \parameter MaximumStepLength: the starting steplength.  \n
   *   example: <tt>(MaxiumStepLength 16.0 8.0 4.0)</tt> \n
   *   Default value: <em>16 / 2^resolutionlevel</em>.
   *
   * \ingroup Optimizers
   */

  template <class TElastix>
    class RSGDEachParameterApart :
    public
      itk::RSGDEachParameterApartOptimizer,
    public
      OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef RSGDEachParameterApart              Self;
    typedef RSGDEachParameterApartOptimizer     Superclass1;
    typedef OptimizerBase<TElastix>             Superclass2;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( RSGDEachParameterApart, RSGDEachParameterApartOptimizer );

    /** Name of this class.
     * Use this name in the parameter file to select this specific optimizer. \n
     * example: <tt>(Optimizer "RSGDEachParameterApart")</tt>\n
     */
    elxClassNameMacro( "RSGDEachParameterApart" );

    /** Typedef's inherited from Superclass1.*/
    typedef Superclass1::CostFunctionType     CostFunctionType;
    typedef Superclass1::CostFunctionPointer  CostFunctionPointer;

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

    /** Methods that have to be present everywhere.*/
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

    RSGDEachParameterApart(){};
      virtual ~RSGDEachParameterApart() {};

  private:

      RSGDEachParameterApart( const Self& );  // purposely not implemented
      void operator=( const Self& );              // purposely not implemented

  };


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxRSGDEachParameterApart.hxx"
#endif

#endif // end #ifndef __elxRSGDEachParameterApart_h
