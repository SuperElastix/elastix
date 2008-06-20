/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxFullSearch2_h
#define __elxFullSearch2_h

#include "itkFullSearchOptimizer2.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

  /**
   * \class FullSearch2
   * \brief An optimizer based on the itk::FullSearchOptimizer2.
   *
   * This class is a wrap around the FullSearchOptimizer2 class.
   * It takes care of setting parameters and printing progress information.
   * For more information about the optimisation method, please read the documentation
   * of the FullSearchOptimizer2 class.
   * 
   * Watch out for this optimizer; it may be very slow....
   *
   * The parameters used in this class are:
   * \parameter Optimizer: Select this optimizer as follows:\n
   *		<tt>(Optimizer "FullSearch2")</tt>
   * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
   *		example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
   *    Default value: 500.
   *
   * \ingroup Optimizers
   * \sa FullSearchOptimizer2
   */

  template <class TElastix>
    class FullSearch2 :
    public
      itk::FullSearchOptimizer2,
    public
      OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef FullSearch2						Self;
    typedef FullSearchOptimizer2	Superclass1;
    typedef OptimizerBase<TElastix>										Superclass2;
    typedef SmartPointer<Self>												Pointer;
    typedef SmartPointer<const Self>									ConstPointer;
    
    /** Method for creation through the object factory. */
    itkNewMacro( Self );
    
    /** Run-time type information (and related methods). */
    itkTypeMacro( FullSearch2, FullSearchOptimizer2 );
    
    /** Name of this class.
     * Use this name in the parameter file to select this specific optimizer. \n
     * example: <tt>(Optimizer "FullSearchOptimizer2")</tt>\n
     */
    elxClassNameMacro( "FullSearch2" );

    /** Typedef's inherited from Superclass1.*/
    typedef Superclass1::CostFunctionType			CostFunctionType;
    typedef Superclass1::CostFunctionPointer	CostFunctionPointer;
    typedef Superclass1::StopConditionType		StopConditionType;
    
    /** Typedef's inherited from Elastix.*/
    typedef typename Superclass2::ElastixType						ElastixType;
    typedef typename Superclass2::ElastixPointer				ElastixPointer;
    typedef typename Superclass2::ConfigurationType			ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
    typedef typename Superclass2::RegistrationType			RegistrationType;
    typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
    typedef typename Superclass2::ITKBaseType						ITKBaseType;
    
    /** Typedef for the ParametersType. */
    typedef typename Superclass1::ParametersType				ParametersType;

    /** Methods that take care of setting parameters and printing progress information.*/
    virtual void BeforeRegistration(void);
    virtual void BeforeEachResolution(void);
    virtual void AfterEachResolution(void);
    virtual void AfterEachIteration(void);
    virtual void AfterRegistration(void);		

    /** Check if any scales are set, and set the UseScales flag on or off; 
     * after that call the superclass' implementation */
    //virtual void StartOptimization(void);
            
  protected:

      FullSearch2();
      virtual ~FullSearch2() {};
      
  private:

      FullSearch2( const Self& );	// purposely not implemented
      void operator=( const Self& );							// purposely not implemented
      
  }; // end class FullSearch2
  

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxFullSearch2.hxx"
#endif

#endif // end #ifndef __elxFullSearch2_h

