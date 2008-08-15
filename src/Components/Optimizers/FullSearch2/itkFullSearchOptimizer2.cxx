/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkFullSearchOptimizer2_cxx
#define __itkFullSearchOptimizer2_cxx

#include "itkFullSearchOptimizer2.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

#include "math.h"
#include "vnl/vnl_math.h"


namespace itk
{


  /**
   * ************************* Constructor ************************
   */

  FullSearchOptimizer2
    ::FullSearchOptimizer2()
  {
    itkDebugMacro( "Constructor" );
    
    this->m_Maximize = false;
    this->m_NumberOfIterations = 100;
    this->m_CurrentIteration = 0;
    this->m_Value = 0.0;
    this->m_BestValue = 0.0;
    this->m_StopCondition = FullRangeSearched;
    this->m_Step.resize( 0 );
    
  } // end Constructor
  
  
  /**
   * ************************* PrintSelf **************************
   */

  void
    FullSearchOptimizer2
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
    
    os << indent << "NumberOfIterations: "
      << this->m_NumberOfIterations << std::endl;
    os << indent << "CurrentIteration: "
      << this->m_CurrentIteration;
    os << indent << "Value: "
      << this->m_Value;
    os << indent << "StopCondition: "
      << this->m_StopCondition;
    os << std::endl;
    
  } // end PrintSelf
  
  
  /**
   * *********************** StartOptimization ********************
   */

  void
    FullSearchOptimizer2
    ::StartOptimization(void)
  {   
    itkDebugMacro( "StartOptimization" );
    
    this->m_CurrentIteration = 0;
    this->m_Stop = false;

    /** Set the initial position */
    const unsigned int spaceDimension = 
      this->GetCostFunction()->GetNumberOfParameters();
    ParametersType initialPosition( spaceDimension );
    for ( unsigned int j = 0; j < spaceDimension; j++ )
    {
      initialPosition[ j ] = this->m_Step[ 0 ] * m_BasePosition[ j ];
    }
    this->SetCurrentPosition( initialPosition );

    /** Initialize best value. */
    if (m_Maximize)
    {
      this->m_BestValue = NumericTraits<double>::NonpositiveMin();
    }
    else
    {
      this->m_BestValue = NumericTraits<double>::max();
    }

    /** Resume. */
    if ( !this->m_Stop )
    {
      this->ResumeOptimization();
    }
      
  } // end StartOptimization  
  
  
  /**
   * ********************** ResumeOptimization ********************
   */

  void
    FullSearchOptimizer2
    ::ResumeOptimization( void )
  {   
    itkDebugMacro( "ResumeOptimization" );
    
    this->m_Stop = false;
    
    InvokeEvent( StartEvent() );
    while ( ! this->m_Stop ) 
    {   
      /** Compute the current value. */
      try
      {
        this->m_Value = this->m_CostFunction->GetValue( this->GetCurrentPosition() );
      }
      catch( ExceptionObject& err )
      {
        // An exception has occurred. 
        // Terminate immediately.
        this->m_StopCondition = MetricError;
        StopOptimization();
          
        // Pass exception to caller
        throw err;
      }
      if ( m_Stop )
      {
        break;
      }
      /** Check if the value is a minimum or maximum */
      // ^ = xor, yields true if only one of the expressions is true
      if ( ( this->m_Value < this->m_BestValue ) ^ this->m_Maximize )
      {
        this->m_BestValue = this->m_Value;
        //m_BestPointInSearchSpace = m_CurrentPointInSearchSpace;
        //m_BestIndexInSearchSpace = m_CurrentIndexInSearchSpace;
      }

      this->InvokeEvent( IterationEvent() );

      /** Prepare for next step */
      this->m_CurrentIteration++;
      
      if( this->m_CurrentIteration >= this->m_NumberOfIterations )
      {
        this->m_StopCondition = FullRangeSearched;
        StopOptimization();
        break;
      }

      this->AdvanceOneStep();
      
    } // while !m_stop
        
  } // end ResumeOptimization
  
  
  /**
   * ********************** StopOptimization **********************
   */

  void
    FullSearchOptimizer2
    ::StopOptimization( void )
  {   
    itkDebugMacro( "StopOptimization" );
    
    this->m_Stop = true;
    //this->SetCurrentPosition( this->GetCurrentPosition() );
    InvokeEvent( EndEvent() );

  } // end StopOptimization
  
  
  /**
   * ********************** AdvanceOneStep ************************
   */

  void
    FullSearchOptimizer2
    ::AdvanceOneStep( void )
  {   
    itkDebugMacro( "AdvanceOneStep" );
    
    double step = this->m_Step[ this->m_CurrentIteration ];

    const unsigned int spaceDimension = 
      this->GetCostFunction()->GetNumberOfParameters();
    
    ParametersType newPosition( spaceDimension );
    for ( unsigned int j = 0; j < spaceDimension; j++ )
    {
      newPosition[ j ] = step * m_BasePosition[ j ];
    }
    
    this->SetCurrentPosition( newPosition );
    
    //this->InvokeEvent( IterationEvent() );
    
  } // end AdvanceOneStep
  
    
} // end namespace itk


#endif // end #ifndef __itkFullSearchOptimizer2_cxx

