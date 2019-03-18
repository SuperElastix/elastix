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
#ifndef _itkPreconditionedGradientDescentOptimizer_cxx
#define _itkPreconditionedGradientDescentOptimizer_cxx

#include "itkPreconditionedGradientDescentOptimizer.h"

#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"
#include "vnl/vnl_math.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_sparse_symmetric_eigensystem.h"

namespace itk
{
/** Error handler for cholmod */
static void my_cholmod_handler( int status, const char *file, int line,
  const char *message )
{
  std::cerr << "cholmod error: file: " << file << "line: " << line
    << "status: " << status << ": " << message << std::endl;
  //\todo: write to xout, throw exception
  //itkGenericExceptionMacro( << "Cholmod error - file: " << file << "line: " << line << "status: " << status << ": " << message );
}


/**
 * ****************** Constructor ************************
 */

PreconditionedGradientDescentOptimizer
::PreconditionedGradientDescentOptimizer()
{
  itkDebugMacro("Constructor");

  this->m_LearningRate = 1.0;
  this->m_NumberOfIterations = 100;
  this->m_CurrentIteration = 0;
  this->m_Value = 0.0;
  this->m_StopCondition = MaximumNumberOfIterations;
  this->m_DiagonalWeight = 1e-6;
  this->m_MinimumGradientElementMagnitude = 1e-10;
  this->m_LargestEigenValue = 1.0;
  this->m_Sparsity = 1.0;
  this->m_ConditionNumber = 1.0;

  /** Prepare cholmod */
  this->m_CholmodCommon = new cholmod_common;
  if( this->m_CholmodCommon )
  {
    cholmod_start( this->m_CholmodCommon );
    this->m_CholmodCommon->error_handler = my_cholmod_handler;

    /** We do not plan to modify the factorization. */
    this->m_CholmodCommon->grow0 = 0;
    this->m_CholmodCommon->grow2 = 0;

    /** Use LL' decomposition, not LDL'
     * 0 = FALSE, 1 = TRUE, gcc does not know these macros, so just use 1.
     * A boolean 'true' would probably also work fine.
     */
    this->m_CholmodCommon->final_ll = 1;
  }

  this->m_CholmodFactor = 0;
  this->m_CholmodGradient = 0;

} // end Constructor


/**
 * ****************** Destructor ************************
 */

PreconditionedGradientDescentOptimizer
::~PreconditionedGradientDescentOptimizer()
{
  if( this->m_CholmodCommon )
  {
    if( this->m_CholmodFactor )
    {
      cholmod_free_factor( &this->m_CholmodFactor, this->m_CholmodCommon );
      this->m_CholmodFactor = 0;
    }

    if( this->m_CholmodGradient )
    {
      cholmod_free_sparse( &this->m_CholmodGradient, this->m_CholmodCommon );
      this->m_CholmodGradient = 0;
    }

    cholmod_finish( this->m_CholmodCommon );

    delete this->m_CholmodCommon;
    this->m_CholmodCommon = 0;
  }

} // end Destructor


/**
 * *************** PrintSelf *************************
 */

void
PreconditionedGradientDescentOptimizer
::PrintSelf( std::ostream & os, Indent indent ) const
{
  this->Superclass::PrintSelf( os, indent );

  os << indent << "LearningRate: " << this->m_LearningRate << std::endl;
  os << indent << "NumberOfIterations: " << this->m_NumberOfIterations << std::endl;
  os << indent << "CurrentIteration: " << this->m_CurrentIteration << std::endl;
  os << indent << "Value: " << this->m_Value << std::endl;
  os << indent << "StopCondition: " << this->m_StopCondition << std::endl;
  os << indent << "Gradient: " << this->m_Gradient << std::endl;

} // end PrintSelf()


/**
 * **************** StartOptimization ********************
 */

void
PreconditionedGradientDescentOptimizer
::StartOptimization( void )
{
  this->m_CurrentIteration = 0;

  /** Get the number of parameters; checks also if a cost function has been set at all.
   * if not: an exception is thrown.
   */
  this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Initialize the scaledCostFunction with the currently set scales */
  this->InitializeScales();

  /** Set the current position as the scaled initial position */
  this->SetCurrentPosition( this->GetInitialPosition() );

  this->ResumeOptimization();
} // end StartOptimization()


/**
 * ************************ ResumeOptimization *************
 */

void
PreconditionedGradientDescentOptimizer
::ResumeOptimization( void )
{
  itkDebugMacro("ResumeOptimization");

  this->m_Stop = false;

  InvokeEvent( StartEvent() );

  const unsigned int spaceDimension
    = this->GetScaledCostFunction()->GetNumberOfParameters();
  this->m_Gradient = DerivativeType( spaceDimension );   // check this

  while( ! this->m_Stop )
  {
    try
    {
      this->GetScaledValueAndDerivative(
        this->GetScaledCurrentPosition(), this->m_Value, this->m_Gradient );
    }
    catch( ExceptionObject& err )
    {
      this->MetricErrorResponse( err );
    }

    /** StopOptimization may have been called. */
    if( this->m_Stop )
    {
      break;
    }

    this->AdvanceOneStep();

    /** StopOptimization may have been called. */
    if( this->m_Stop )
    {
      break;
    }

    this->m_CurrentIteration++;

    if( this->m_CurrentIteration >= this->m_NumberOfIterations )
    {
      this->m_StopCondition = MaximumNumberOfIterations;
      this->StopOptimization();
      break;
    }

  } // end while

} // end ResumeOptimization()


/**
 * ***************** MetricErrorResponse ************************
 */

void
PreconditionedGradientDescentOptimizer
::MetricErrorResponse( ExceptionObject & err )
{
  /** An exception has occurred. Terminate immediately. */
  this->m_StopCondition = MetricError;
  this->StopOptimization();

  /** Pass exception to caller. */
  throw err;

} // end MetricErrorResponse()


/**
 * ***************** StopOptimization ************************
 */

void
PreconditionedGradientDescentOptimizer
::StopOptimization( void )
{
  itkDebugMacro("StopOptimization");

  this->m_Stop = true;
  this->InvokeEvent( EndEvent() );
} // end StopOptimization()


/**
 * ************ AdvanceOneStep ****************************
 */

void
PreconditionedGradientDescentOptimizer
::AdvanceOneStep( void )
{
  typedef DerivativeType::ValueType      DerivativeValueType;
  typedef DerivativeType::const_iterator DerivativeIteratorType;

  const unsigned int spaceDimension =
    this->GetScaledCostFunction()->GetNumberOfParameters();

  const ParametersType & currentPosition = this->GetScaledCurrentPosition();
  DerivativeType & searchDirection = this->m_SearchDirection;

  /** Compute the search direction */
  this->CholmodSolve( this->m_Gradient, searchDirection );

  /** Compute the new position */
  ParametersType newPosition( spaceDimension );
  for( unsigned int j = 0; j < spaceDimension; ++j )
  {
    newPosition[ j ] = currentPosition[ j ] - this->m_LearningRate * searchDirection[ j ];
  }

  this->SetScaledCurrentPosition( newPosition );

  this->InvokeEvent( IterationEvent() );

} // end AdvanceOneStep()


/**
 * ************ CholmodSolve ****************************
 */

void
PreconditionedGradientDescentOptimizer
::CholmodSolve( const DerivativeType & gradient,
  DerivativeType & searchDirection, int solveType )
{
  /** This function uses m_CholmodGradient, m_CholmodCommon and m_CholmodFactor,
   * and is therefore not thread-safe.
   */
  itkDebugMacro("CholmodSolve");

  typedef DerivativeType::ValueType      DerivativeValueType;
  typedef DerivativeType::const_iterator DerivativeIteratorType;

  /** Get the spaceDimension from the cost function. */
  const unsigned int spaceDimension =
    this->GetScaledCostFunction()->GetNumberOfParameters();

  if( ! this->m_CholmodFactor )
  {
    /** This shouldn't happen of course
     * Should we throw an error? Or return some error code?
     */
    searchDirection = gradient;
    return;
  }

  searchDirection.SetSize( spaceDimension );
  searchDirection.Fill( 0.0 );

  /** Copy gradient to cholmodGradient */
  const double minimumGradientElementMagnitude =
    this->m_MinimumGradientElementMagnitude;

  /** size nnz */
  CInt * gRow = reinterpret_cast<CInt *>( this->m_CholmodGradient->i );
  double * gVal = reinterpret_cast<double *>( this->m_CholmodGradient->x );

  /** size 1+1 (1 column only) */
  CInt * gCol = reinterpret_cast<CInt *>( this->m_CholmodGradient->p );
  gCol[ 0 ] = 0;
  gCol[ 1 ] = 0;
  DerivativeIteratorType derivIt = gradient.begin();
  const DerivativeIteratorType derivBegin = gradient.begin();
  while( derivIt != gradient.end() )
  {
    const double currentVal = (*derivIt);
    if( std::abs(currentVal) > minimumGradientElementMagnitude )
    {
      ++( gCol[1] );
      (*gRow) = static_cast<CInt>(derivIt - derivBegin);
      (*gVal) = *derivIt;
      ++gRow;
      ++gVal;
    }
    ++derivIt;
  }

  /** solve H x = g, where P~H^{-1}; x will be the search direction. */
  cholmod_sparse * x = cholmod_spsolve( solveType, this->m_CholmodFactor,
    this->m_CholmodGradient, this->m_CholmodCommon );

  /** Copy x to searchDirection */
  CInt * xRow = reinterpret_cast<CInt *>( x->i );
  double * xVal = reinterpret_cast<double *>( x->x );

  /** size 1+1 (1 column only) */
  CInt * xCol = reinterpret_cast<CInt *>( x->p );
  const CInt * xRowEnd = xRow + xCol[1];
  for( ; xRow < xRowEnd ; ++xRow, ++xVal )
  {
    searchDirection[ static_cast<unsigned int>(*xRow) ] = (*xVal);
  }

} // end CholmodSolve()


/**
 * ************ SetPreconditionMatrix ****************************
 */

void
PreconditionedGradientDescentOptimizer
::SetPreconditionMatrix( PreconditionType & precondition )
{
  /** Compute and modify eigensystem of preconditioning matrix.
   * Does not take into account scales (yet)!
   */
  itkDebugMacro("SetPreconditionMatrix");

  typedef PreconditionType::row             RowType;
  typedef RowType::const_iterator           RowIteratorType;
  typedef vnl_vector<PreconditionValueType> DiagonalType;

  const size_t spaceDimension = static_cast<size_t>( precondition.cols() );

  /** Count the number of nonzero elements. */
  size_t nnz = 0;

  /** Estimates largest eigenvalue. */
  double maxDiag = 0;

  /** estimate of largest eigenvalue, not very accurate */
  //DiagonalType diagP( spaceDimension );
  //precondition.diag_AtA( diagP );
  //const double frobnormP = std::sqrt( diagP.sum() );

  /** Check range of eigenvalues */
  for( unsigned int r = 0; r < spaceDimension; ++r )
  {
    PreconditionValueType & prr = precondition( r, r );
    maxDiag = vnl_math_max( maxDiag, prr );
  }

  /** make positive definite by adding a small negligible fraction of maxDiag.
   * This does not really affect the maximum eigen value, but seems to make
   * the eig routine a bit more robust.
   */
  const double diagTemp = maxDiag * 1e-3;
  for( unsigned int r = 0; r < spaceDimension; ++r )
  {
    PreconditionValueType & prr = precondition( r, r );
    prr += diagTemp;
  }

  /** Estimate largest eigenvalue to 1 decimal digit precision
   * If eig fails (which it does quite regularly) use the maxDiag value.
   */
  const long ndigits = 1;
  vnl_sparse_symmetric_eigensystem eig;
  int errorCode = eig.CalculateNPairs( precondition, 1, false, ndigits );
  double & largestEig = this->m_LargestEigenValue;
  if( errorCode == 0 )
  {
    largestEig = eig.get_eigenvalue( 0 );
  }
  else
  {
    largestEig = maxDiag;
  }

  /** Subtract diagTemp and add diagWeight * largestEig */
  const double diagDef = this->m_DiagonalWeight * largestEig - diagTemp;
  for( unsigned int r = 0; r < spaceDimension; ++r )
  {
    PreconditionValueType & prr = precondition( r, r );
    prr += diagDef;
    nnz += precondition.get_row(r).size();
  }

  /** Store some information for the user: */
  this->m_Sparsity = static_cast<double>( nnz ) /
    static_cast<double>( spaceDimension * spaceDimension );

  /** Create sparse matrix in cholmod_sparse format. The supplied
   * precondition matrix is symmetric. Only the upper triangular part
   * is stored, in a row-based compressed format. Cholmod adopts a
   * column-based compressed format, so when we will copy this matrix,
   * we implicitly transpose the matrix. The cholmod_sparse will thus
   * have stype -1, meaning that the lower-triangular part is stored.
   */
  const int stype = -1;
  const bool sorted = true;
  const bool packed = true;
  cholmod_sparse * cPrecondition = cholmod_allocate_sparse(
    spaceDimension, spaceDimension, nnz, sorted, packed,
    stype, CHOLMOD_REAL, this->m_CholmodCommon );

  /** size nnz */
  CInt * cRow = reinterpret_cast<CInt *>( cPrecondition->i );
  double * cVal = reinterpret_cast<double *>( cPrecondition->x );
  /** size spaceDimension+1 */
  CInt * cCol = reinterpret_cast<CInt *>( cPrecondition->p );

  /** Loop over rows of input matrix */
  const CInt * cRowBegin = cRow;
  for( unsigned int r = 0; r < spaceDimension; ++r )
  {
    RowType & rowVector = precondition.get_row( r );
    (*cCol) = static_cast<int>( cRow - cRowBegin );
    ++cCol;

    /** Iterate over each row */
    for( RowIteratorType rowIt = rowVector.begin(); rowIt != rowVector.end(); ++rowIt )
    {
      (*cRow) = rowIt->first;
      (*cVal) = rowIt->second;
      ++cRow;
      ++cVal;
    }
  }
  (*cCol) = static_cast<int>(cRow - cRowBegin);

  /** Sanity check */
  if( static_cast<size_t>(*cCol) != nnz )
  {
    /** Release memory */
    cholmod_free_sparse( &cPrecondition, this->m_CholmodCommon );
    itkExceptionMacro( "ERROR: unexpected error during conversion to cholmod format");
  }

  /** Destroy precondition input, to save memory */
  precondition.set_size( 0, 0 );

  /** Prepare for factorization */
  this->m_CholmodFactor = cholmod_analyze( cPrecondition, this->m_CholmodCommon );

  /** Factorize cPrediction + diagonalWeight * largestEig * Identity */
  double beta[2];
  beta[0] = 0.0; // this->GetDiagonalWeight() * largestEig; but we already did that above
  beta[1] = 0.0; // this is for potential imaginary part of complex number.
  cholmod_factorize_p( cPrecondition, beta, NULL, 0,
    this->m_CholmodFactor, this->m_CholmodCommon );

  /** Store condition number of user */
  this->m_ConditionNumber = cholmod_rcond( this->m_CholmodFactor, this->m_CholmodCommon );

  /** Release memory */
  cholmod_free_sparse( &cPrecondition, this->m_CholmodCommon );

  /** Prepare cholmod sparse structure for gradients */
  const int stypeg = 0;
  if( this->m_CholmodGradient )
  {
    cholmod_free_sparse( &this->m_CholmodGradient, this->m_CholmodCommon );
    this->m_CholmodGradient = 0;
  }
  this->m_CholmodGradient = cholmod_allocate_sparse(
    spaceDimension, 1, spaceDimension, sorted, packed,
    stypeg, CHOLMOD_REAL, this->m_CholmodCommon );

} // end SetPreconditionMatrix()


} // end namespace itk

#endif
