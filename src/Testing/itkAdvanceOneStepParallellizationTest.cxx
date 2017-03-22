/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkSmartPointer.h"
#include "itkArray.h"
#include <vector>
#include <algorithm>
#include <iomanip>

// Report timings
#include <ctime>
#include "itkTimeProbe.h"
#include "itkTimeProbesCollectorBase.h"

// Multi-threading using ITK threads
#include "itkMultiThreader.h"

// Multi-threading using OpenMP
#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

// Single-threaded vector arithmetic using Eigen
#ifdef ELASTIX_USE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>
#endif

// select double or float internal type of array
#if 0
typedef float InternalScalarType;
#else
typedef double InternalScalarType;
#endif

#ifdef ELASTIX_USE_EIGEN
#if 0
typedef Eigen::VectorXf ParametersTypeEigen;
#else
typedef Eigen::VectorXd ParametersTypeEigen;
#endif
#endif

class OptimizerTEMP : public itk::Object
{
public:

  /** Standard class typedefs. */
  typedef OptimizerTEMP             Self;
  typedef itk::SmartPointer< Self > Pointer;
  itkNewMacro( Self );

  typedef itk::Array< InternalScalarType > ParametersType;

  unsigned long      m_NumberOfParameters;
  ParametersType     m_CurrentPosition;
  ParametersType     m_Gradient;
  InternalScalarType m_LearningRate;

  typedef itk::MultiThreader             ThreaderType;
  typedef ThreaderType::ThreadInfoStruct ThreadInfoType;
  ThreaderType::Pointer m_Threader;
  bool                  m_UseOpenMP;
  bool                  m_UseEigen;
  bool                  m_UseMultiThreaded;

  struct MultiThreaderParameterType
  {
    ParametersType * t_NewPosition;
    Self *           t_Optimizer;
  };

  OptimizerTEMP()
  {
    this->m_NumberOfParameters = 0;
    this->m_LearningRate       = 0.0;
    this->m_Threader           = ThreaderType::New();
    this->m_Threader->SetNumberOfThreads( 8 );
    this->m_UseOpenMP        = false;
    this->m_UseEigen         = false;
    this->m_UseMultiThreaded = false;
  }


  void AdvanceOneStep( void )
  {
    const unsigned int spaceDimension = m_NumberOfParameters;
    ParametersType &   newPosition    = this->m_CurrentPosition;

    if( !this->m_UseMultiThreaded )
    {
      /** Get a pointer to the current position. */
      const InternalScalarType * currentPosition = this->m_CurrentPosition.data_block();
      const double               learningRate    = this->m_LearningRate;
      const InternalScalarType * gradient        = this->m_Gradient.data_block();
      InternalScalarType *       newPos          = newPosition.data_block();

      /** Update the new position. */
      for( unsigned int j = 0; j < spaceDimension; j++ )
      {
        //newPosition[j] = currentPosition[j] - this->m_LearningRate * this->m_Gradient[j];
        newPos[ j ] = currentPosition[ j ] - learningRate * gradient[ j ];
      }
    }
#ifdef ELASTIX_USE_OPENMP
    else if( this->m_UseOpenMP && !this->m_UseEigen )
    {
      /** Get a pointer to the current position. */
      const InternalScalarType * currentPosition = this->m_CurrentPosition.data_block();
      const InternalScalarType   learningRate    = this->m_LearningRate;
      const InternalScalarType * gradient        = this->m_Gradient.data_block();
      InternalScalarType *       newPos          = newPosition.data_block();

      /** Update the new position. */
      const int nthreads = static_cast< int >( this->m_Threader->GetNumberOfThreads() );
      omp_set_num_threads( nthreads );
      #pragma omp parallel for
      for( int j = 0; j < static_cast< int >( spaceDimension ); j++ )
      {
        newPos[ j ] = currentPosition[ j ] - learningRate * gradient[ j ];
      }
    }
#endif
#ifdef ELASTIX_USE_EIGEN
    else if( !this->m_UseOpenMP && this->m_UseEigen )
    {
      /** Get a reference to the current position. */
      const ParametersType &   currentPosition = this->m_CurrentPosition;
      const InternalScalarType learningRate    = this->m_LearningRate;

      /** Wrap itk::Arrays into Eigen jackets. */
      Eigen::Map< ParametersTypeEigen >       newPositionE( newPosition.data_block(), spaceDimension );
      Eigen::Map< const ParametersTypeEigen > currentPositionE( currentPosition.data_block(), spaceDimension );
      Eigen::Map< ParametersTypeEigen >       gradientE( this->m_Gradient.data_block(), spaceDimension );

      /** Update the new position. */
      //Eigen::setNbThreads( this->m_Threader->GetNumberOfThreads() );
      newPositionE = currentPositionE - learningRate * gradientE;
    }
#endif
    else
    {
      /** Fill the threader parameter struct with information. */
      MultiThreaderParameterType * temp = new  MultiThreaderParameterType;
      temp->t_NewPosition = &newPosition;
      temp->t_Optimizer   = this;

      /** Call multi-threaded AdvanceOneStep(). */
      this->m_Threader->SetSingleMethod( AdvanceOneStepThreaderCallback, (void *)( temp ) );
      this->m_Threader->SingleMethodExecute();

      delete temp;
    }
  }  // end


  /** The callback function. */
  static ITK_THREAD_RETURN_TYPE AdvanceOneStepThreaderCallback( void * arg )
  {
    /** Get the current thread id and user data. */
    ThreadInfoType *             infoStruct = static_cast< ThreadInfoType * >( arg );
    itk::ThreadIdType            threadID   = infoStruct->ThreadID;
    MultiThreaderParameterType * temp
      = static_cast< MultiThreaderParameterType * >( infoStruct->UserData );

    /** Call the real implementation. */
    temp->t_Optimizer->ThreadedAdvanceOneStep2( threadID, *( temp->t_NewPosition ) );

    return ITK_THREAD_RETURN_VALUE;

  }  // end AdvanceOneStepThreaderCallback()


  /** The threaded implementation of AdvanceOneStep(). */
  inline void ThreadedAdvanceOneStep( itk::ThreadIdType threadId, ParametersType & newPosition )
  {
    /** Compute the range for this thread. */
    const unsigned int spaceDimension = m_NumberOfParameters;
    const unsigned int subSize        = static_cast< unsigned int >(
      vcl_ceil( static_cast< double >( spaceDimension )
      / static_cast< double >( this->m_Threader->GetNumberOfThreads() ) ) );
    const unsigned int jmin = threadId * subSize;
    unsigned int       jmax = ( threadId + 1 ) * subSize;
    jmax = ( jmax > spaceDimension ) ? spaceDimension : jmax;

    /** Get a reference to the current position. */
    const ParametersType & currentPosition = this->m_CurrentPosition;
    const double           learningRate    = this->m_LearningRate;
    const ParametersType & gradient        = this->m_Gradient;

    /** Advance one step: mu_{k+1} = mu_k - a_k * gradient_k */
    for( unsigned int j = jmin; j < jmax; j++ )
    {
      newPosition[ j ] = currentPosition[ j ] - learningRate * gradient[ j ];
    }

  }  // end ThreadedAdvanceOneStep()


  /** The threaded implementation of AdvanceOneStep(). */
  inline void ThreadedAdvanceOneStep2( itk::ThreadIdType threadId, ParametersType & newPosition )
  {
    /** Compute the range for this thread. */
    const unsigned int spaceDimension = m_NumberOfParameters;
    const unsigned int subSize        = static_cast< unsigned int >(
      vcl_ceil( static_cast< double >( spaceDimension )
      / static_cast< double >( this->m_Threader->GetNumberOfThreads() ) ) );
    const unsigned int jmin = threadId * subSize;
    unsigned int       jmax = ( threadId + 1 ) * subSize;
    jmax = ( jmax > spaceDimension ) ? spaceDimension : jmax;

    /** Get a pointer to the current position. */
    const InternalScalarType * currentPosition = this->m_CurrentPosition.data_block();
    const double               learningRate    = this->m_LearningRate;
    const InternalScalarType * gradient        = this->m_Gradient.data_block();
    InternalScalarType *       newPos          = newPosition.data_block();

    /** Advance one step: mu_{k+1} = mu_k - a_k * gradient_k */
    for( unsigned int j = jmin; j < jmax; j++ )
    {
      newPos[ j ] = currentPosition[ j ] - learningRate * gradient[ j ];
    }

  }  // end ThreadedAdvanceOneStep()


};

// end class Optimizer

//-------------------------------------------------------------------------------------

int
main( int argc, char * argv[] )
{
  // Declare and setup
  std::cout << std::fixed << std::showpoint << std::setprecision( 8 );
  std::cout << "RESULTS FOR InternalScalarType = " << typeid( InternalScalarType ).name()
            << "\n\n" << std::endl;

  /** Typedefs. */
  typedef OptimizerTEMP                  OptimizerClass;
  typedef OptimizerClass::ParametersType ParametersType;

  OptimizerClass::Pointer optimizer = OptimizerClass::New();

  // test parameters
  std::vector< unsigned int > arraySizes;
  arraySizes.push_back( 1e2 ); arraySizes.push_back( 1e3 ); arraySizes.push_back( 1e4 );
  arraySizes.push_back( 1e5 ); arraySizes.push_back( 1e6 ); arraySizes.push_back( 1e7 );
  std::vector< unsigned int > repetitions;
  repetitions.push_back( 2e7 ); repetitions.push_back( 2e6 ); repetitions.push_back( 2e5 );
  repetitions.push_back( 2e4 ); repetitions.push_back( 1e3 ); repetitions.push_back( 1e2 );

  /** For all sizes. */
  for( unsigned int s = 0; s < arraySizes.size(); ++s )
  {
    std::cout << "Array size = " << arraySizes[ s ] << std::endl;

    /** Setup. */
    itk::TimeProbesCollectorBase timeCollector;
    repetitions[ s ] = 1; // outcomment this line for full testing

    ParametersType newPos( arraySizes[ s ] );
    ParametersType curPos( arraySizes[ s ] );
    ParametersType gradient( arraySizes[ s ] );
    for( unsigned int i = 0; i < arraySizes[ s ]; ++i )
    {
      curPos[ i ]   = 2.1;
      gradient[ i ] = 2.1;
    }
    optimizer->m_NumberOfParameters = arraySizes[ s ];
    optimizer->m_LearningRate       = 3.67;
    optimizer->m_CurrentPosition    = curPos;
    optimizer->m_Gradient           = gradient;

    /** Time the ITK single-threaded implementation. */
    optimizer->m_UseOpenMP        = false;
    optimizer->m_UseEigen         = false;
    optimizer->m_UseMultiThreaded = false;
    for( unsigned int i = 0; i < repetitions[ s ]; ++i )
    {
      timeCollector.Start( "st" );
      optimizer->AdvanceOneStep();
      timeCollector.Stop( "st" );
    }

    /** Time the ITK multi-threaded implementation. */
    optimizer->m_UseOpenMP        = false;
    optimizer->m_UseEigen         = false;
    optimizer->m_UseMultiThreaded = true;
    unsigned int rep = repetitions[ s ] / 1000.0;
    if( rep < 10 ) { rep = 10; }
    for( unsigned int i = 0; i < rep; ++i )
    {
      timeCollector.Start( "ITK (mt)" );
      optimizer->AdvanceOneStep();
      timeCollector.Stop( "ITK (mt)" );
    }

    /** Time the OpenMP multi-threaded implementation. */
#ifdef ELASTIX_USE_OPENMP
    optimizer->m_UseOpenMP        = true;
    optimizer->m_UseEigen         = false;
    optimizer->m_UseMultiThreaded = true;
    if( arraySizes[ s ] < 10000 )
    {
      rep = repetitions[ s ] / 100.0;
      if( rep < 10 ) { rep = 10; }
    }
    else { rep = repetitions[ s ]; }
    for( unsigned int i = 0; i < rep; ++i )
    {
      timeCollector.Start( "OMP (mt)" );
      optimizer->AdvanceOneStep();
      timeCollector.Stop( "OMP (mt)" );
    }
#endif

    /** Time the Eigen single-threaded implementation. */
#ifdef ELASTIX_USE_EIGEN
    optimizer->m_UseOpenMP        = false;
    optimizer->m_UseEigen         = true;
    optimizer->m_UseMultiThreaded = true;
    for( unsigned int i = 0; i < repetitions[ s ]; ++i )
    {
      timeCollector.Start( "Eigen (st)" );
      optimizer->AdvanceOneStep();
      timeCollector.Stop( "Eigen (st)" );
    }
#endif

    // Report timings for this array size
    timeCollector.Report( std::cout, false, true );
    std::cout << std::endl;

  } // end loop over array sizes

  return EXIT_SUCCESS;

} // end main
