/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkFullSearchOptimizer2_h
#define __itkFullSearchOptimizer2_h

#include "itkSingleValuedNonLinearOptimizer.h"


namespace itk
{

  /**
   * \class FullSearchOptimizer2
   * \brief An optimizer 
   *
   * \ingroup Optimizers
   * \sa FiniteDifferenceGradientDescent
   */
  
  class FullSearchOptimizer2
    : public SingleValuedNonLinearOptimizer
  {
  public:
    
    /** Standard class typedefs. */
    typedef FullSearchOptimizer2		            Self;
    typedef SingleValuedNonLinearOptimizer	   	Superclass;
    typedef SmartPointer<Self>									Pointer;
    typedef SmartPointer<const Self>						ConstPointer;
    
    /** Method for creation through the object factory. */
    itkNewMacro( Self );
    
    /** Run-time type information (and related methods). */
    itkTypeMacro( FullSearchOptimizer2, SingleValuedNonLinearOptimizer );

    typedef Superclass::ParametersType								ParametersType;
    
    /** Codes of stopping conditions */
    typedef enum {
        FullRangeSearched,
        MetricError
    } StopConditionType;
      
    /** Advance one step following the gradient direction. */
    virtual void AdvanceOneStep( void );
    
    /** Start optimization. */
    void StartOptimization( void );
    
    /** Resume previously stopped optimization with current parameters
    * \sa StopOptimization. */
    void ResumeOptimization( void );
    
    /** Stop optimization.
    * \sa ResumeOptimization */
    void StopOptimization( void );
    
    /** Get the number of iterations. */
    itkGetConstMacro( NumberOfIterations, unsigned long );
    
    /** Get the current iteration number. */
    itkGetConstMacro( CurrentIteration, unsigned long );
    
    /** Get the current value. */
    itkGetConstMacro( Value, double );
    itkGetConstMacro( BestValue, double );
    
    /** Get Stop condition. */
    itkGetConstMacro( StopCondition, StopConditionType );
  
    void SetStep( const std::vector< double > & steps )
    {
      this->m_Step = steps;
      this->m_NumberOfIterations = steps.size();
    }

    //itkGetMacro( Step, std::vector< double > );
    double GetStep( unsigned int i ) const
    {
      return this->m_Step[ i ];
    }

    itkSetMacro( BasePosition, ParametersType );

    
  protected:

    FullSearchOptimizer2();
    virtual ~FullSearchOptimizer2() {};

    /** PrintSelf method.*/
    void PrintSelf( std::ostream& os, Indent indent ) const;

    bool                          m_Maximize;
    bool                          m_Stop;
    double                        m_Value;
    double												m_BestValue;
    StopConditionType             m_StopCondition;
    std::vector< double >         m_Step;
    ParametersType                m_BasePosition;
        
  private:

    FullSearchOptimizer2( const Self& );	// purposely not implemented
    void operator=( const Self& );										// purposely not implemented
    
    /** Private member variables.*/
    unsigned long                 m_NumberOfIterations;
    unsigned long                 m_CurrentIteration;
    
  }; // end class FullSearchOptimizer2


} // end namespace itk


#endif // end #ifndef __itkFullSearchOptimizer2_h

