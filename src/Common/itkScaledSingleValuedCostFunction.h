#ifndef __itkScaledSingleValuedCostFunction_h
#define __itkScaledSingleValuedCostFunction_h

#include "itkSingleValuedCostFunction.h"

namespace itk
{
  /** 
   * \class ScaledSingleValuedCostFunction
   * \brief A cost function that applies a scaling to another cost function
   *
   * This class can be used to adapt an existing, badly scaled, cost function. 
   * 
   * By default it does not apply any scaling. Use the method SetUseScales(true)
   * to enable the use of scales.
   *
   * \ingroup Numerics
   */

  class ScaledSingleValuedCostFunction : public SingleValuedCostFunction
  {
  public:

		/** Standard ITK-stuff. */
    typedef ScaledSingleValuedCostFunction  Self;
    typedef SingleValuedCostFunction        Superclass;
    typedef SmartPointer<Self>              Pointer;
    typedef SmartPointer<const Self>        ConstPointer;

		/** Method for creation through the object factory. */
    itkNewMacro( Self );

		/** Run-time type information (and related methods). */
    itkTypeMacro( ScaledSingleValuedCostFunction, SingleValuedCostFunction );

		/** Typedefs inherited from the superclass. */
    typedef Superclass::MeasureType         MeasureType;
    typedef Superclass::DerivativeType      DerivativeType;
    typedef Superclass::ParametersType      ParametersType;
    typedef Superclass::Pointer             SingleValuedCostFunctionPointer;

    typedef Array<double>                   ScalesType;

    /** Divide the parameters by the scales and call the GetValue routine 
     * of the unscaled cost function.
		 */
    virtual MeasureType GetValue(
      const ParametersType & parameters ) const;

    /** Divide the parameters by the scales, call the GetDerivative routine
     * of the unscaled cost function and divide the resulting derivative by
     * the scales.
		 */
    virtual void GetDerivative(
      const ParametersType & parameters,
      DerivativeType & derivative ) const;

    /** Same procedure as in GetValue and GetDerivative. */
    virtual void GetValueAndDerivative(
      const ParametersType & parameters,
      MeasureType & value,
      DerivativeType & derivative ) const;

    /** Ask the UnscaledCostFunction how many parameters it has. */
    virtual unsigned int GetNumberOfParameters(void) const;

    /** Set the cost function that needs scaling. */
    itkSetObjectMacro( UnscaledCostFunction, Superclass );
		/** Get the cost function that needs scaling. */
    itkGetObjectMacro( UnscaledCostFunction, Superclass );

    /** Set the scales. */
    virtual void SetScales( const ScalesType & scales );
		/** Get the scales. */
    itkGetConstReferenceMacro( Scales, ScalesType );

    /** Set the flag to use scales or not. */
    itkSetMacro( UseScales, bool );
		/** Get the flag to use scales or not. */
    itkGetConstMacro( UseScales, bool );

		/** Set the flag to negate the cost function or not. */
    itkBooleanMacro( NegateCostFunction );
		/** Set the flag to negate the cost function or not. */
    itkSetMacro( NegateCostFunction, bool );
		/** Get the flag to negate the cost function or not. */
    itkGetConstMacro( NegateCostFunction, bool );

    /** Convert the parameters from scaled to unscaled: x = y/s. */
    virtual void ConvertScaledToUnscaledParameters( ParametersType & parameters ) const;

		/** Convert the parameters from unscaled to scaled: y = x*s. */
    virtual void ConvertUnscaledToScaledParameters( ParametersType & parameters ) const;

  protected:

		/** The constructor. */
    ScaledSingleValuedCostFunction();
		/** The destructor. */
    virtual ~ScaledSingleValuedCostFunction() {};

		/** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const{};

  private:

		/** The private constructor. */
    ScaledSingleValuedCostFunction( const Self& );	// purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );									// purposely not implemented

		/** Member variables. */
    ScalesType                            m_Scales;
    SingleValuedCostFunctionPointer       m_UnscaledCostFunction;
    bool                                  m_UseScales;      
    bool                                  m_NegateCostFunction;
    

  }; // end class ScaledSingleValuedCostFunction

} //end namespace itk


#endif // #ifndef __itkScaledSingleValuedCostFunction_h

