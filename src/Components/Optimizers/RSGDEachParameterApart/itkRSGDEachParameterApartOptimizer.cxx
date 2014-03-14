/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkRSGDEachParameterApartOptimizer_cxx
#define __itkRSGDEachParameterApartOptimizer_cxx

#include "itkRSGDEachParameterApartOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"

namespace itk
{

/**
 * Advance one Step following the gradient direction
 * This method will be overrided in non-vector spaces
 */
void
RSGDEachParameterApartOptimizer
::StepAlongGradient( const DerivativeType & factor,
  const DerivativeType & transformedGradient )
{

  itkDebugMacro( << "factor = " << factor << "  transformedGradient= " << transformedGradient );

  const unsigned int spaceDimension
    = m_CostFunction->GetNumberOfParameters();

  ParametersType newPosition( spaceDimension );
  ParametersType currentPosition = this->GetCurrentPosition();

  for( unsigned int j = 0; j < spaceDimension; j++ )
  {
    /** Each parameters has its own factor! */
    newPosition[ j ] = currentPosition[ j ] + transformedGradient[ j ] * factor[ j ];
  }

  itkDebugMacro( << "new position = " << newPosition );

  this->SetCurrentPosition( newPosition );

}


} // end namespace itk

#endif
