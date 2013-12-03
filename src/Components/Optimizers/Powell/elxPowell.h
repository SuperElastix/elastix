/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxPowell_h
#define __elxPowell_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkPowellOptimizer.h"

namespace elastix
{

  /**
   * \class Powell
   * \brief An optimizer based on Powell...
   *
   * This optimizer is a wrap around the itk::PowellOptimizer.
   * This wrap-around class takes care of setting parameters, and printing progress
   * information.
   * For detailed information about the optimisation method, please read the
   * documentation of the itkPowellOptimizer (in the ITK-manual).
   * \sa ImprovedPowellOptimizer
   * \ingroup Optimizers
   */

  template <class TElastix>
    class Powell :
    public
      itk::PowellOptimizer,
    public
      OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef Powell                                   Self;
    typedef PowellOptimizer                          Superclass1;
    typedef OptimizerBase<TElastix>                  Superclass2;
    typedef itk::SmartPointer<Self>                  Pointer;
    typedef itk::SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( Powell, PowellOptimizer );

    /** Name of this class.
     * Use this name in the parameter file to select this specific optimizer. \n
     * example: <tt>(Optimizer "Powell")</tt>\n
     */
    elxClassNameMacro( "Powell" );

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

    /** Methods invoked by elastix, in which parameters can be set and
     * progress information can be printed. */
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

    Powell(){};
    virtual ~Powell() {};

  private:

    Powell( const Self& );  // purposely not implemented
    void operator=( const Self& );              // purposely not implemented

  };


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxPowell.hxx"
#endif

#endif // end #ifndef __elxPowell_h
