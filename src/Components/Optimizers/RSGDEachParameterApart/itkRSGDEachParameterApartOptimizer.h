/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkRSGDEachParameterApartOptimizer_h
#define __itkRSGDEachParameterApartOptimizer_h

#include "itkRSGDEachParameterApartBaseOptimizer.h"

namespace itk
{

  /**
   * \class RSGDEachParameterApartOptimizer
   * \brief An optimizer based on gradient descent.
   *
   * This class is almost a copy of the normal
   * itk::RegularStepGradientDescentOptimizer. The difference
   * is that each parameter has its own step length, whereas the normal
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
   * Note that this is a quite experimental optimizer, currently
   * only used for some specific tests.
   *
   * \ingroup Optimizers
   * \sa RSGDEachParameterApart
   */

  class RSGDEachParameterApartOptimizer :
    public RSGDEachParameterApartBaseOptimizer
  {
  public:
    /** Standard class typedefs. */
    typedef RSGDEachParameterApartOptimizer         Self;
    typedef RSGDEachParameterApartBaseOptimizer     Superclass;
    typedef SmartPointer<Self>                          Pointer;
    typedef SmartPointer<const Self>                    ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( RSGDEachParameterApartOptimizer,
      RSGDEachParameterApartBaseOptimizer );

    /** Cost function typedefs. */
    typedef Superclass::CostFunctionType        CostFunctionType;
    typedef CostFunctionType::Pointer           CostFunctionPointer;

  protected:

    RSGDEachParameterApartOptimizer() {};
    virtual ~RSGDEachParameterApartOptimizer() {};

    /** Advance one step along the corrected gradient taking into
    * account the steplengths represented by the factor array.
    * This method is invoked by AdvanceOneStep. It is expected
    * to be overrided by optimization methods in non-vector spaces
    * \sa AdvanceOneStep */
    virtual void StepAlongGradient(
      const DerivativeType & factor,
      const DerivativeType & transformedGradient );

  private:

    RSGDEachParameterApartOptimizer( const Self& ); // purposely not implemented
    void operator=( const Self& );                  // purposely not implemented

  };

} // end namespace itk


#endif // end #ifndef __itkRSGDEachParameterApartOptimizer_h



